"""경험 버퍼 모음.

강화학습 알고리즘은 크게 두 부류이고, 쓰는 버퍼도 다르다:

  · on-policy  (PPO, A2C, ...)  → RolloutBuffer
      "지금 정책"으로 모은 경험만 쓸 수 있다.
      업데이트가 끝나면 데이터를 전부 버리고 새로 모은다.

  · off-policy (DQN, SAC, ...)  → ReplayBuffer
      과거 정책이 모은 경험도 재사용할 수 있다.
      큰 순환 버퍼에 계속 쌓아두고 무작위로 뽑아 학습한다.

버퍼를 utils/로 분리한 이유: 버퍼는 알고리즘의 "부품"이지 본체가 아니다.
train.py(메인)에서 버퍼를 만들어 에이전트에 주입(dependency injection)하면,
- 어떤 버퍼를 쓰는지가 메인에서 한눈에 보이고
- 나중에 버퍼를 바꾸거나(예: 우선순위 리플레이) 다른 알고리즘을 붙일 때
  알고리즘 코드를 열지 않아도 된다.
"""
import random
from collections import deque

import numpy as np
import torch


# ======================================================================
# On-policy용: RolloutBuffer  (PPO가 사용)
# ======================================================================
class RolloutBuffer:
    """한 번의 rollout(n_steps) 동안의 경험을 담아두는 버퍼.

    PPO는 on-policy 알고리즘이라, 업데이트 후에는 이 데이터를 버리고
    새 정책으로 다시 수집한다 (ReplayBuffer처럼 오래 쌓아두지 않는다).

    저장하는 것: (obs, action, log_prob, reward, value, done)
        log_prob : 그 행동을 뽑을 당시의 log π_old(a|s).
                   PPO의 확률비 r(θ) 계산에 필요해서 함께 저장한다.
        value    : 그 상태의 V(s) 추정치. GAE 계산에 필요.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(float(done))

    def __len__(self):
        return len(self.rewards)

    # ------------------------------------------------------------------
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """GAE(λ)로 advantage와 return(critic 타깃)을 계산한다.

            δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
            A_t = δ_t + (γλ)·(1-done)·A_{t+1}     ← 뒤에서 앞으로 재귀
            R_t = A_t + V(s_t)

        Args:
            last_value: rollout 마지막 상태의 V(s). 에피소드가 중간에
                끊긴 경우 그 이후의 가치를 부트스트랩하는 데 쓴다.
        Returns:
            (advantages, returns) — 둘 다 np.ndarray, 길이 T
        """
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        T = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]     # 종료 상태면 미래가치 차단
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def to_tensors(self, device):
        """(obs, actions, old_log_probs)를 학습용 텐서로 변환."""
        return (
            torch.as_tensor(np.asarray(self.obs, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=device),
        )


# ======================================================================
# Off-policy용: ReplayBuffer  (DQN, SAC 등을 추가할 때 사용)
#
# 현재 PPO는 이 버퍼를 쓰지 않지만, 새 알고리즘을 붙일 때
# 그대로 가져다 쓸 수 있도록 표준형을 만들어둔다.
# ======================================================================
class ReplayBuffer:
    """고정 크기 순환(circular) 경험 버퍼.

    (s, a, r, s', done) 전이를 capacity까지 쌓고, 가득 차면
    가장 오래된 것부터 밀어낸다. sample()로 무작위 미니배치를 뽑는다.
    무작위 샘플링은 연속된 전이 사이의 상관관계를 깨서 학습을 안정화한다.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, float(done)))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int, device=None):
        """무작위 미니배치를 텐서로 반환: (obs, act, rew, next_obs, done)"""
        batch = self.rng.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.asarray, zip(*batch))
        to = lambda x, dt=np.float32: torch.as_tensor(
            np.asarray(x, dtype=dt), device=device)
        return to(obs), to(act), to(rew), to(next_obs), to(done)
