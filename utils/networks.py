"""신경망 부품 모음.

알고리즘(algorithms/)과 분리해둔 이유: 네트워크 구조는 알고리즘과 독립적이다.
PPO든 SAC든 A2C든 "관측을 받아 행동분포와 가치를 내놓는 몸통"은 공유할 수 있다.
새 알고리즘을 만들 때 여기서 필요한 부품만 가져다 쓰면 된다.

제공하는 것:
    build_mlp(...)          : 범용 MLP 생성 함수
    GaussianActorCritic     : 연속 행동용 actor-critic (PPO가 사용)
    CategoricalActorCritic  : 이산 행동용 actor-critic (참고/확장용)
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


# ======================================================================
# 범용 MLP 빌더
# ======================================================================
def build_mlp(in_dim: int, hidden_sizes, out_dim: int = None,
              activation: str = "tanh") -> nn.Sequential:
    """[in_dim] → hidden_sizes → (out_dim) 형태의 MLP를 만든다.

    Args:
        in_dim:        입력 차원
        hidden_sizes:  은닉층 크기 리스트. 예: [64, 64]
        out_dim:       출력 차원. None이면 마지막 은닉층까지만 만든다
                       (= 여러 head가 공유하는 "몸통/트렁크"로 쓸 때)
        activation:    "tanh" | "relu" | "elu"
                       tanh는 소규모 RL에서 관례적으로 안정적이고,
                       relu는 층이 깊거나 큰 문제에서 유리한 경향.

    Returns:
        nn.Sequential. 마지막 층은 활성함수 없음 (out_dim을 준 경우).
    """
    act_layer = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

    layers, last = [], in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), act_layer()]
        last = h
    if out_dim is not None:
        layers.append(nn.Linear(last, out_dim))   # 출력층엔 활성함수 X
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = 1.0):
    """직교 초기화 — PPO 논문 계열 구현에서 표준으로 쓰이는 초기화.

    policy head는 작은 gain(0.01)으로 초기화해서 학습 초기에
    모든 행동이 거의 균등하게 나오도록 만드는 것이 관례다
    (특정 행동으로 성급하게 쏠리는 것을 방지).
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    return module


# ======================================================================
# 연속 행동용 Actor-Critic (Gaussian 정책)
#
#   π(a|s) = N( μ(s), σ )
#     μ(s) : 신경망 출력. tanh로 [-1,1]에 묶는다 (환경의 행동 범위와 일치)
#     σ    : 상태와 무관한 학습 파라미터 log_std → σ = exp(log_std)
#            학습 초반엔 크게(탐험), 진행되며 작아진다(정밀 제어)
#
#   critic V(s)는 별도 head. trunk를 공유해 파라미터를 절약한다.
# ======================================================================
class GaussianActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64),
                 activation: str = "tanh", init_log_std=-0.5,
                 squash_mean: bool = True):
        """
        Args:
            init_log_std: log_std 초기값 (스칼라 또는 차원별 시퀀스).
                σ = exp(init_log_std).
                -0.5 → σ ≈ 0.61.  행동 범위가 [-1,1]이므로 σ=1.0(기본 0)은
                사실상 완전 랜덤에 가까워 학습이 잘 안 굳는다. 그래서
                조금 작은 값에서 시작하는 편이 안정적이다.
            squash_mean: True면 μ에 tanh를 적용해 [-1,1]로 제한.
                평균이 범위 밖으로 나가면 샘플이 항상 클리핑에 걸려
                gradient 정보가 죽으므로 켜두는 것을 권장.
        """
        super().__init__()
        self.squash_mean = squash_mean

        self.trunk = build_mlp(obs_dim, hidden_sizes, None, activation)
        last = hidden_sizes[-1]
        self.mu_head = orthogonal_init(nn.Linear(last, act_dim), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)
        # init_log_std: 스칼라(모든 차원 동일) 또는 차원별 시퀀스.
        #   예) (-0.5, 0.0) → 가감속 σ=e^-0.5≈0.61, 차선변경 σ=e^0=1.0
        # 차원별로 다르게 주는 이유: 행동마다 "탐험의 가치"가 다르다.
        #   차선변경처럼 양자화 임계(|raw|>0.5)를 넘어야만 효과가 있는
        #   차원은 σ를 크게 시작해야 임계를 자주 넘어 그 이득(추월)을
        #   경험할 수 있다. 학습이 진행되면 각 차원의 σ는 독립적으로
        #   경사하강으로 조정된다.
        if np.isscalar(init_log_std):
            init = torch.ones(act_dim) * float(init_log_std)
        else:
            assert len(init_log_std) == act_dim, \
                f"init_log_std 길이 {len(init_log_std)} != act_dim {act_dim}"
            init = torch.as_tensor([float(x) for x in init_log_std])
        self.log_std = nn.Parameter(init)

    def forward(self, obs):
        """Returns: (mu, std, value)"""
        z = self.trunk(obs)
        mu = self.mu_head(z)
        if self.squash_mean:
            mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        return mu, std, self.v_head(z).squeeze(-1)

    # ---- 알고리즘이 쓰는 표준 인터페이스 -------------------------------
    def distribution(self, obs) -> Normal:
        mu, std, _ = self.forward(obs)
        return Normal(mu, std)

    def evaluate_actions(self, obs, actions):
        """저장된 행동에 대한 (log_prob, entropy, value)를 반환.

        PPO 업데이트에서 확률비 r(θ) = exp(logp_new - logp_old)를
        계산할 때 쓰인다. 다차원 행동은 차원별 독립 가정 하에 합산.
        """
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def current_std(self) -> float:
        """로깅용: 현재 탐험 폭(σ)의 평균."""
        return float(torch.exp(self.log_std).mean().item())


# ======================================================================
# 이산 행동용 Actor-Critic (Categorical 정책)
#   현재 환경은 연속 행동을 쓰므로 사용하지 않지만,
#   행동을 이산으로 바꾸는 실험을 할 때 그대로 쓸 수 있게 남겨둔다.
# ======================================================================
class CategoricalActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(64, 64),
                 activation: str = "tanh"):
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_sizes, None, activation)
        last = hidden_sizes[-1]
        self.pi_head = orthogonal_init(nn.Linear(last, n_actions), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)

    def forward(self, obs):
        z = self.trunk(obs)
        return self.pi_head(z), self.v_head(z).squeeze(-1)

    def distribution(self, obs) -> Categorical:
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()), dist.entropy(), value

    @property
    def current_std(self) -> float:
        return float("nan")   # 이산 정책엔 σ 개념이 없음


# ======================================================================
# 하이브리드 Actor-Critic: 연속 가감속 + "이산" 차선변경   ★ 권장 정책 ★
#
# 왜 하이브리드인가 — 양자화 dead zone 문제:
#   차선변경을 연속값으로 내고 |raw|>0.5에서 자르면, (-0.5,0.5) 구간은
#   출력이 뭐든 아무 일도 안 일어나 gradient가 노이즈다. 게다가
#   deterministic 평가는 평균 μ만 쓰므로, μ가 0 근처에 머무는 한
#   변경이 영원히 0회가 된다 (실측: 탐험 σ=1.0으로 키워도 μ_lane이
#   0에 고정되어 eval 차선변경 0회 지속).
#
# 해결: 차선변경을 3-way Categorical(왼쪽/유지/오른쪽)로 분리.
#   - 각 선택의 확률이 명시적 → gradient가 깨끗하다 (dead zone 없음)
#   - deterministic 평가 = argmax → 학습된 만큼 변경이 "실제로 나온다"
#   - 탐험 = 확률 그 자체 (keep_bias로 초기 시도율을 직관적으로 제어)
#
# 행동 벡터 인터페이스는 기존과 동일하게 유지한다:
#   action = [가감속 실수, 차선명령 실수(-1/0/+1)]
#   → env의 양자화(_quantize_lane_change)가 ±1을 그대로 통과시키므로
#     환경 코드는 수정 없이 동작한다.
#
# log π(a|s) = Normal.log_prob(가감속) + Categorical.log_prob(차선)
#   (두 행동을 조건부 독립으로 두는 표준 구성)
# ======================================================================
class HybridActorCritic(nn.Module):
    # 차선 인덱스 ↔ 명령 매핑: 0=왼쪽(+1), 1=유지(0), 2=오른쪽(-1)
    LANE_CMDS = (1.0, 0.0, -1.0)

    def __init__(self, obs_dim: int, hidden_sizes=(64, 64),
                 activation: str = "tanh", init_log_std=-0.5,
                 lane_keep_bias: float = 1.0):
        """
        Args:
            init_log_std: "가감속" 축의 초기 탐험 폭 σ = exp(·).
                (시퀀스가 오면 첫 원소를 사용 — Gaussian 정책과의 호환용)
            lane_keep_bias: 차선 head의 "유지" 로짓 초기 편향.
                1.0 → 초기 확률 약 (좌 16%, 유지 68%, 우 16%)
                    = 스텝당 ~32%가 변경 시도 (탐험 충분 + 과폭주 방지)
                0.0 → 균등 1/3씩 (~67% 시도), 2.0 → ~14% 시도
        """
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden_sizes, None, activation)
        last = hidden_sizes[-1]
        self.mu_head = orthogonal_init(nn.Linear(last, 1), gain=0.01)
        self.lane_head = orthogonal_init(nn.Linear(last, 3), gain=0.01)
        self.v_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)
        with torch.no_grad():
            self.lane_head.bias[1] = lane_keep_bias   # "유지"를 기본값으로

        if not np.isscalar(init_log_std):
            init_log_std = float(init_log_std[0])
        self.log_std = nn.Parameter(torch.ones(1) * float(init_log_std))

    def forward(self, obs):
        z = self.trunk(obs)
        mu = torch.tanh(self.mu_head(z))          # 가감속 평균 ∈ [-1,1]
        std = torch.exp(self.log_std)
        lane_logits = self.lane_head(z)
        return mu, std, lane_logits, self.v_head(z).squeeze(-1)

    # ---- 알고리즘이 쓰는 표준 인터페이스 ----------------------------
    def sample(self, obs):
        """행동 샘플. Returns (action[2], log_prob, value) — 배치 없는 단일 관측용."""
        mu, std, logits, value = self.forward(obs)
        nd = Normal(mu, std)
        accel = nd.sample()
        lane_idx = Categorical(logits=logits).sample()
        log_prob = (nd.log_prob(accel).sum(-1)
                    + Categorical(logits=logits).log_prob(lane_idx))
        lane_cmd = torch.as_tensor(
            [self.LANE_CMDS[int(i)] for i in lane_idx], device=accel.device)
        action = torch.cat([accel, lane_cmd.unsqueeze(-1)], dim=-1)
        return action, log_prob, value

    def det_action(self, obs):
        """결정적 행동: 가감속=μ, 차선=argmax. 학습됐다면 여기서 변경이 나온다."""
        mu, _, logits, _ = self.forward(obs)
        lane_idx = torch.argmax(logits, dim=-1)
        lane_cmd = torch.as_tensor(
            [self.LANE_CMDS[int(i)] for i in lane_idx], device=mu.device)
        return torch.cat([mu, lane_cmd.unsqueeze(-1)], dim=-1)

    def evaluate_actions(self, obs, actions):
        """저장된 행동의 (log_prob, entropy, value). PPO 업데이트용.

        actions[:,0]=가감속, actions[:,1]=차선명령(-1/0/+1) → 인덱스 복원.
        """
        mu, std, logits, value = self.forward(obs)
        nd = Normal(mu, std)
        cd = Categorical(logits=logits)
        accel = actions[:, :1]
        # 명령(-1/0/+1) → 인덱스(2/1/0):  idx = 1 - round(cmd)
        lane_idx = (1.0 - torch.round(actions[:, 1])).long().clamp(0, 2)
        log_prob = nd.log_prob(accel).sum(-1) + cd.log_prob(lane_idx)
        entropy = nd.entropy().sum(-1) + cd.entropy()
        return log_prob, entropy, value

    def lane_entropy(self, obs) -> torch.Tensor:
        """차선 head(Categorical)만의 엔트로피. 차선 전용 탐험 보너스용.

        전체 엔트로피(가감속+차선)에 계수를 걸면 가감속 σ까지 안 줄어드는
        부작용이 있으므로(과거 entropy_coef=0.01 수렴실패 사례), 차선
        분포에만 별도 계수를 거는 용도로 분리해둔다.
        """
        _, _, logits, _ = self.forward(obs)
        return Categorical(logits=logits).entropy()

    def lane_change_prob(self, obs) -> torch.Tensor:
        """진단용: 상태별 '변경(좌+우)' 확률. 로깅에 사용."""
        _, _, logits, _ = self.forward(obs)
        p = torch.softmax(logits, dim=-1)
        return 1.0 - p[:, 1]

    @property
    def current_std(self) -> float:
        return float(torch.exp(self.log_std).mean().item())
