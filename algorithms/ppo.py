"""PPO(Proximal Policy Optimization) — 알고리즘 본체.

이 파일에는 "알고리즘"만 있다:
    네트워크 구조  → utils/networks.py
    경험 버퍼      → utils/buffer.py   (RolloutBuffer — 메인에서 주입받음)
    정책 평가      → utils/evaluator.py
    로깅          → utils/logger.py
덕분에 아래 코드는 PPO 수식과 1:1로 대응해서 읽을 수 있다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PPO 요약 (Schulman et al., 2017)

policy gradient의 기본:
    advantage A가 양수인 행동의 확률은 높이고, 음수면 낮춘다.

문제: 한 번의 업데이트로 정책이 너무 크게 변하면 학습이 붕괴한다.
해법: 새/옛 정책의 확률비를 클리핑해서 과도한 업데이트의 이득을 제거.

    r(θ) = π_θ(a|s) / π_θ_old(a|s)

    L^CLIP(θ) = E[ min( r(θ)·A ,  clip(r(θ), 1-ε, 1+ε)·A ) ]

전체 손실:
    L = -L^CLIP  +  c_v · L^VF  -  c_e · H[π]
        ────────     ─────────      ───────
        정책 개선     가치 회귀      탐험 유지
    (L^VF = (V(s) - R_target)², H = 정책 엔트로피)

advantage는 GAE(λ)로 추정:
    δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    A_t = δ_t + (γλ)·(1-done)·A_{t+1}          ← 뒤에서 앞으로 재귀
    R_t = A_t + V(s_t)                          ← critic의 회귀 타깃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

학습 루프:
    반복 {
      1) rollout 수집   : 현재 정책으로 n_steps만큼 환경과 상호작용
      2) advantage 계산 : GAE(λ)
      3) 정책 업데이트  : 같은 데이터로 n_epochs번 미니배치 경사하강
      4) (주기적) 평가  : 노이즈 없는 정책으로 주행 지표 측정
    }
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from utils.networks import GaussianActorCritic
from utils.evaluator import evaluate_policy
from utils.buffer import RolloutBuffer


# ======================================================================
class PPO:
    """PPO 에이전트.

    하이퍼파라미터는 생성자 인자로 명시적으로 받는다
    (train.py에서 한눈에 보고 바꿀 수 있도록).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        # ---- 최적화 ----
        lr: float = 3e-4,
        n_steps: int = 2048,          # rollout 길이 (업데이트 1회당 수집량)
        n_epochs: int = 10,           # 같은 데이터 재사용 횟수
        minibatch_size: int = 64,
        # ---- 할인 / advantage ----
        gamma: float = 0.99,          # 할인율
        gae_lambda: float = 0.95,     # GAE λ
        # ---- PPO 고유 ----
        clip_eps: float = 0.2,        # 클리핑 범위 ε
        value_coef: float = 0.5,      # 가치 손실 가중치 c_v
        entropy_coef: float = 0.0,    # 엔트로피 보너스 c_e
        max_grad_norm: float = 0.5,   # gradient clipping
        normalize_advantage: bool = True,
        # ---- 네트워크 ----
        hidden_sizes=(64, 64),
        activation: str = "tanh",
        init_log_std: float = -0.5,   # 초기 탐험 폭 σ = exp(init_log_std)
        # ---- 기타 ----
        seed: int = 0,
        device: str = "auto",
        buffer: RolloutBuffer = None,   # 경험 버퍼 (메인에서 주입).
                                        # None이면 내부에서 기본 생성.
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 네트워크는 utils/networks.py 에서 가져온다
        self.policy = GaussianActorCritic(
            obs_dim, act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            init_log_std=init_log_std,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage

        # 버퍼 주입(dependency injection):
        # train.py(메인)가 utils.buffer.RolloutBuffer를 만들어 넘겨준다.
        # 어떤 버퍼를 쓰는지가 메인에서 보이고, 나중에 다른 버퍼로 교체할 때
        # 이 파일을 수정할 필요가 없다.
        self.buffer = buffer if buffer is not None else RolloutBuffer()
        self.num_timesteps = 0

    # ==================================================================
    # 행동 선택
    # ==================================================================
    @torch.no_grad()
    def act(self, obs):
        """학습용: 정책 분포에서 샘플 (σ가 탐험을 담당).

        Returns: (action, log_prob, value)
            log_prob과 value는 나중에 PPO 업데이트/GAE에 필요하므로 함께 반환.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        mu, std, value = self.policy(obs_t)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return (action.squeeze(0).cpu().numpy().astype(np.float32),
                float(log_prob.item()), float(value.item()))

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        """평가/실행용: deterministic이면 분포의 평균 μ를 그대로 사용.

        utils/evaluator.py 가 요구하는 인터페이스이기도 하다.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        mu, std, _ = self.policy(obs_t)
        if deterministic:
            return mu.squeeze(0).cpu().numpy().astype(np.float32)
        return (torch.normal(mu, std).squeeze(0)
                .cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def get_value(self, obs) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        _, _, value = self.policy(obs_t)
        return float(value.item())

    # ==================================================================
    # 1단계) rollout 수집
    # ==================================================================
    def collect_rollout(self, env, state, logger=None, ep_stats=None):
        """현재 정책으로 n_steps만큼 경험을 모은다.

        Args:
            state: 직전 rollout이 끝난 시점의 관측 (에피소드를 이어서 수집)
            ep_stats: [ep_return, ep_length] 리스트 (호출 간 누적 유지)
        Returns:
            (last_obs, last_value, finished_returns)
        """
        obs = state
        finished_returns = []
        self.buffer.clear()

        for _ in range(self.n_steps):
            action, log_prob, value = self.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            self.num_timesteps += 1
            ep_stats[0] += reward
            ep_stats[1] += 1

            # ★ truncation(시간 초과) 보정:
            #   MDP가 끝난 게 아니라 시간 제한으로 잘린 것이므로,
            #   이어졌다면 받았을 미래 가치를 V(s')로 근사해 보상에 더한다.
            #   생략하면 "시간초과 = 이후 가치 0"으로 잘못 학습된다.
            stored_reward = reward
            if truncated and not terminated:
                stored_reward += self.gamma * self.get_value(next_obs)

            self.buffer.add(obs, action, log_prob, stored_reward, value,
                            terminated)

            if terminated or truncated:
                finished_returns.append(ep_stats[0])
                if logger is not None:
                    logger.log_episode(self.num_timesteps, ep_stats[0],
                                       int(ep_stats[1]))
                ep_stats[0], ep_stats[1] = 0.0, 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # rollout이 에피소드 중간에 끊겼다면 이후 가치를 부트스트랩
        return obs, self.get_value(obs), finished_returns

    # ==================================================================
    # 3단계) 정책 업데이트 (clipped surrogate)
    # ==================================================================
    def update(self, advantages, returns) -> dict:
        obs_t, act_t, old_log_prob_t = self.buffer.to_tensors(self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)

        # advantage 정규화: 스케일을 평균0/표준편차1로 → 학습률 민감도 감소
        if self.normalize_advantage:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T = len(self.buffer)
        indices = np.arange(T)
        stats = {"policy_loss": 0.0, "value_loss": 0.0,
                 "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_batches = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.minibatch_size):
                mb = indices[start:start + self.minibatch_size]

                # 현재 정책으로 "그때 그 행동"을 다시 평가
                log_prob, entropy, value = self.policy.evaluate_actions(
                    obs_t[mb], act_t[mb])

                # 확률비 r(θ) = exp(log π_new - log π_old)
                ratio = torch.exp(log_prob - old_log_prob_t[mb])

                # ---- PPO 핵심: clipped surrogate objective ----
                surrogate_1 = ratio * adv_t[mb]
                surrogate_2 = torch.clamp(ratio, 1 - self.clip_eps,
                                          1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # ---- critic: V(s)를 GAE 타깃에 회귀 ----
                value_loss = ((value - ret_t[mb]) ** 2).mean()

                # ---- 엔트로피 보너스: 탐험 유지 ----
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                # 진단 지표
                with torch.no_grad():
                    # approx_kl: 정책이 얼마나 변했는지. 0.02를 크게 넘으면
                    #   업데이트가 과격하다는 신호 (lr을 낮출 것)
                    approx_kl = (old_log_prob_t[mb] - log_prob).mean()
                    # clip_frac: 클리핑에 걸린 샘플 비율. 0.1~0.3이 보통
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl.item()
                stats["clip_frac"] += clip_frac.item()
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in stats.items()}

    # ==================================================================
    # 메인 학습 루프
    # ==================================================================
    def learn(self, env, total_timesteps: int, logger=None,
              log_interval: int = 1, eval_interval: int = 0,
              eval_episodes: int = 5):
        """학습 실행.

        Args:
            logger: utils.logger.RunLogger (없으면 콘솔 출력만)
            eval_interval: 몇 번의 업데이트마다 평가할지 (0이면 평가 안 함)
        """
        obs, _ = env.reset()
        ep_stats = [0.0, 0]                    # [현재 에피소드 보상, 길이]
        recent_returns = deque(maxlen=20)
        n_updates = 0

        while self.num_timesteps < total_timesteps:
            # 1) rollout
            obs, last_value, finished = self.collect_rollout(
                env, obs, logger=logger, ep_stats=ep_stats)
            recent_returns.extend(finished)

            # 2) advantage
            advantages, returns = self.buffer.compute_gae(
                last_value, self.gamma, self.gae_lambda)

            # 3) update
            stats = self.update(advantages, returns)
            n_updates += 1

            # 로깅
            metrics = {
                "update": n_updates,
                "ep_rew_mean": float(np.mean(recent_returns))
                               if recent_returns else float("nan"),
                "std": self.policy.current_std,
                **stats,
            }
            if logger is not None:
                logger.log_train(self.num_timesteps, metrics,
                                 verbose=(n_updates % log_interval == 0))

            # 4) 주기적 평가 (utils/evaluator.py 사용)
            if eval_interval > 0 and n_updates % eval_interval == 0:
                m = evaluate_policy(self, env, n_episodes=eval_episodes)
                if logger is not None:
                    logger.log_eval(self.num_timesteps, m.as_dict(),
                                    m.summary(), eval_episodes)
                # 평가가 학습 중이던 에피소드를 끊었으므로 리셋.
                # (끊긴 경험은 이미 버퍼에 반영됐고 GAE가 부트스트랩하므로 무해)
                obs, _ = env.reset()
                ep_stats[0], ep_stats[1] = 0.0, 0

    # ==================================================================
    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device))
