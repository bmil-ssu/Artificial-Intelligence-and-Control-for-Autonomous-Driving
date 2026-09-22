"""단일 시점의 부분관측으로 행동 벡터를 직접 회귀하는 BC 정책.

Behavior Cloning(행동 모방)은 수집한 관측과 행동의 짝을 정답 데이터로 삼아
관측 → 행동 관계를 지도학습하는 방법이다. 이 파일은 신경망과 추론 및 저장을
담당하고, 손실 계산과 가중치 갱신은 외부 학습 코드에서 수행한다.

기본 구조: 관측 → 정규화 → 은닉층 256 → 은닉층 256 → 행동 2개
기본 행동: [가감속 raw, 차선변경 raw]
두 행동을 모두 연속값으로 출력하며, 차선변경을 분류하는 별도 head는 없다.
관측 이력을 기억하는 구조가 아니므로 현재 시점의 관측만으로 행동을 결정한다.
"""

import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """PyTorch가 가중치를 관리하고 역전파할 수 있는 행동 예측 신경망."""

    def __init__(self, state_dim, action_dim=2):
        """state_dim은 관측 길이, action_dim은 출력 행동 벡터의 길이이다."""
        # nn.Module 초기화: 아래에서 정의하는 층과 buffer를 등록할 준비를 한다.
        super().__init__()

        # 여기서 state는 SUMO의 전체 내부 상태가 아니라 에이전트의 부분관측이다.
        # 입력 검증과 모델 복원에 사용하므로 입출력 차원을 보관한다.
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 완전연결층(Linear)은 입력에 학습 가능한 가중치와 편향을 적용한다.
        # fc1: 관측을 256개 특징으로 변환.
        # fc2: 특징 사이의 관계를 조합하여 행동 결정에 필요한 표현을 학습.
        # fc3: 최종 특징을 action_dim개의 행동 값으로 변환.
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        # 관측 항목별 평균과 표준편차. 학습 코드에서 학습 데이터의 통계로 설정한다.
        # 초기값 0과 1은 정규화를 하더라도 입력이 그대로 유지되도록 한다.
        # buffer는 학습으로 갱신되는 파라미터가 아니지만 state_dict에 저장되고,
        # model.to(device)를 호출하면 신경망 가중치와 함께 장치를 이동한다.
        self.register_buffer("obs_mean", torch.zeros(state_dim))
        self.register_buffer("obs_std", torch.ones(state_dim))

    def forward(self, state):
        """단일 관측 (state_dim,) 또는 관측 배치 (N, state_dim)의 행동을 계산."""
        # 학습과 추론에 동일한 통계를 사용하여 입력 항목의 스케일을 맞춘다.
        # 배치 입력에서는 관측 항목별 통계가 각 행에 자동으로 적용된다.
        # obs_std가 0이 되지 않도록 통계를 설정하는 학습 코드에서 처리해야 한다.
        state = (state - self.obs_mean) / self.obs_std

        # ReLU는 음수를 0으로 바꾸는 비선형 함수이다.
        # 층 사이에 비선형성을 넣어 단순한 선형 관계보다 복잡한 행동을 학습한다.
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        # 출력에는 tanh나 clipping이 없으므로 [-1, 1] 범위가 보장되지 않는다.
        # 환경에 행동을 전달할 때 필요한 범위 제한과 차선 명령 양자화를 적용한다.
        return self.fc3(x)

    # 추론에서는 역전파가 필요 없으므로 gradient 기록을 꺼 메모리 사용을 줄인다.
    @torch.no_grad()
    def predict(self, state, deterministic=True):
        """단일 관측을 받아 환경에 전달할 NumPy 행동 벡터를 반환한다.

        deterministic은 기존 정책 평가 코드와 인터페이스를 맞추기 위한 인자다.
        이 모델은 확률분포에서 행동을 뽑지 않으므로 False여도 같은 방식으로 예측한다.
        """
        # 평가 모드로 전환한다. eval() 자체가 gradient 계산을 끄는 것은 아니다.
        # 현재 구조에는 Dropout/BatchNorm이 없지만 추론 의도를 명확히 표시한다.
        self.eval()

        # 리스트나 NumPy 입력을 신경망 가중치와 같은 float32 자료형으로 통일한다.
        state = np.asarray(state, dtype=np.float32)

        # predict는 단일 관측 전용이다. 배치 학습은 forward를 직접 사용한다.
        # 관측 차원이 달라지면 엉뚱한 입력으로 추론하지 않도록 즉시 오류를 낸다.
        if state.shape != (self.state_dim,):
            raise ValueError(
                f"observation shape은 ({self.state_dim},)이어야 합니다."
            )

        # 입력도 모델과 같은 CPU/GPU 장치에 올려 장치 불일치 오류를 방지한다.
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.obs_mean.device,
        )

        # nn.Module의 호출 경로를 통해 forward를 실행한다.
        action = self(state)

        # 환경은 NumPy 배열을 받으므로 CPU로 옮긴 뒤 변환한다.
        # 여기서는 출력값을 제한하지 않고 신경망의 회귀 결과를 그대로 반환한다.
        return action.cpu().numpy()

    def save(self, path):
        """모델 구조를 복원할 차원 정보, 가중치, 정규화 통계를 함께 저장한다."""
        # state_dict에는 fc1~fc3의 가중치/편향뿐 아니라 obs_mean/obs_std도 포함된다.
        # optimizer 상태는 저장하지 않으므로 학습 재개용 전체 상태 저장은 아니다.
        torch.save(
            {
                "algorithm": "bc",
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """저장 파일로부터 새 BCPolicy 객체를 만들어 추론 가능한 상태로 반환한다."""
        # classmethod이므로 기존 객체 없이 BCPolicy.load(path)로 호출할 수 있다.
        # map_location을 지정하면 GPU에서 저장한 가중치도 CPU 등으로 읽을 수 있다.
        # weights_only=True는 텐서와 기본 자료형 중심으로 역직렬화를 제한한다.
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        # 저장 당시의 입력/출력 차원으로 동일한 신경망 구조를 먼저 만든다.
        model = cls(
            checkpoint["state_dim"],
            checkpoint["action_dim"],
        ).to(device)

        # 초기 무작위 가중치를 학습된 값으로 교체하고 정규화 통계도 복원한다.
        model.load_state_dict(checkpoint["state_dict"])
        # 불러온 직후 바로 추론할 수 있도록 평가 모드로 설정한다.
        model.eval()

        return model
