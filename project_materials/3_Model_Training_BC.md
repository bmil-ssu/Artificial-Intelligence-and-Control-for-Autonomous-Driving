# 3. Model Training — Imitation Learning

> **목표:** IDM 기반 일반 차량의 주행 데이터를 수집하고,  
> 해당 데이터를 이용해 자율주행 차량의 의사결정 모델 학습

---

# 1. 목표

앞 단계에서는 SUMO 도로 환경을 만들고,

- Observation
- Action
- Reward

를 정의했습니다.

이번에는 **일반 차량(HV)의 주행 데이터를 이용하여 자율주행 차량(AV)의 Policy를 학습**합니다.

전체 흐름

```text
IDM 차량 주행
      ↓
주행 데이터 수집
      ↓
(Observation, Action) Dataset
      ↓
Behavior Cloning
      ↓
학습된 Policy
      ↓
SUMO에서 AV 주행 평가
```

---

# 2. Behavior Cloning이란?

**Behavior Cloning (BC)** 은 전문가의 행동을 그대로 따라 하도록 학습하는 가장 기본적인 **Imitation Learning** 방법입니다.

이번 프로젝트에서는 SUMO의 IDM 차량을 Expert처럼 사용할 수 있습니다.

예를 들어 IDM 차량이

```text
Observation
=
내 속도
주변 차량과의 상대 위치
주변 차량과의 상대 속도
도로 정보
```

를 보고

```text
Action
=
Acceleration
Lane Change
```

을 수행했다면,

BC 모델은 다음 관계를 학습합니다.

```text
Observation
      ↓
Neural Network
      ↓
Expert Action
```

즉,

> **"이 상황에서 IDM 차량은 어떤 행동을 했는가?"**

를 데이터로부터 배우는 방식입니다.

---

# 3. 왜 BC를 먼저 하나요?

BC는 강화학습보다 구조가 단순합니다.

이미 수집된 Dataset을 이용해

```text
Observation → Action
```

관계를 지도학습으로 학습합니다.

따라서

- Environment가 정상적으로 만들어졌는지 확인하기 쉽고
- Dataset이 올바르게 수집되는지 확인할 수 있고
- 간단하게 기본 Policy를 만들 수 있습니다.

---

# 4. Dataset 수집

일반 차량을 SUMO에서 주행시키면서 매 timestep마다 필요한 값을 저장합니다.

기본적으로 다음 두 정보가 가장 중요합니다.

```text
Observation
Action
```

예:

| Step | Observation | Action |
|---:|---|---|
| 0 | `[0.41, 0.72, ...]` | `[0.31, 0]` |
| 1 | `[0.43, 0.66, ...]` | `[0.12, 0]` |
| 2 | `[0.45, 0.41, ...]` | `[-0.25, -1]` |

필요하다면 이후 Offline RL을 위해 다음 값도 함께 저장할 수 있습니다.

```text
Observation
Action
Reward
Next Observation
Done
```

즉,

```text
(o_t, a_t, r_t, o_(t+1), done)
```

형태로 저장해 두면 이후에도 다시 사용할 수 있습니다.

---

# 5. 가장 중요한 점: Action 정의를 맞추세요

Dataset에 저장하는 Action과 학습 모델이 출력하는 Action의 의미가 같아야 합니다.

예를 들어 환경에서 Action을

```text
Acceleration
Lane Change
```

로 정의했다면 Dataset에서도 동일하게 저장해야 합니다.

```text
Environment Action
        =
Dataset Action
        =
Model Output
```

예를 들어

```text
Acceleration
a_acc ∈ [-1, 1]

Lane Change
a_lc ∈ {-1, 0, +1}
```

이라면 Dataset에도 두 값을 모두 저장합니다.

---

# 6. Dataset 예시

예를 들어 하나의 샘플을 다음과 같이 구성할 수 있습니다.

```python
sample = {
    "observation": [
        ego_speed,
        ego_position,
        relative_speed,
        relative_position,
        density,
        lane,
    ],
    "acceleration": expert_acceleration,
    "lane_change": expert_lane_change,
}
```

여러 timestep에서 이 값을 저장하면 하나의 Dataset이 됩니다.

```text
Episode 1
  ├─ Step 0
  ├─ Step 1
  ├─ Step 2
  └─ ...

Episode 2
  ├─ Step 0
  ├─ Step 1
  └─ ...
```

저장 형식은 자유롭게 선택할 수 있습니다.

예:

- `.npy` (주로 사용)
- `.csv`
- `.pkl`
- `.pt`

중요한 것은 **나중에 다시 불러와 학습할 수 있는 형태로 저장하는 것**입니다.

---

# 7. BC 모델

가장 기본적인 BC 모델은 MLP를 사용할 수 있습니다.

```text
Observation
    ↓
   MLP
    ↓
 Action
```

예:

```python
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)
```

> **단순한 MLP로도 전체 BC Pipeline이 정상적으로 동작하는지 확인하는 것이 먼저입니다.**

---

# 8. Action에 따른 Loss

어떤 Loss를 사용할지는 Action의 형태에 따라 달라집니다.

## Acceleration

Acceleration이 연속적인 값이라면 Regression 문제로 볼 수 있습니다.

예:

```text
Expert acceleration = 0.42
Model acceleration  = 0.37
```

보통 **MSE Loss**를 사용할 수 있습니다.

```python
loss_acc = F.mse_loss(
    pred_acceleration,
    expert_acceleration,
)
```

---

## Lane Change

Lane Change가

```text
Left
Keep
Right
```

와 같은 discrete action이라면 Classification 문제로 볼 수 있습니다.

예:

```text
Left  = 0
Keep  = 1
Right = 2
```

보통 **Cross Entropy Loss**를 사용할 수 있습니다.

```python
loss_lane = F.cross_entropy(
    pred_lane_logits,
    expert_lane_action,
)
```

---

# 9. Hybrid Action을 사용하는 경우

이번 프로젝트처럼

```text
Acceleration + Lane Change
```

를 함께 사용하는 경우에는 모델을 두 개의 Output Head로 구성할 수 있습니다.

```text
                 Observation
                      ↓
                     MLP
                      ↓
             Shared Representation
                /            \
               /              \
              ↓                ↓
      Acceleration Head   Lane Change Head
          Regression       Classification
```

Loss는 예를 들어 다음과 같이 구성할 수 있습니다.

```python
loss = loss_acc + loss_lane
```

필요하다면 각 Loss의 중요도를 조절할 수도 있습니다.

```python
loss = w_acc * loss_acc + w_lane * loss_lane
```

---

# 10. BC Training

학습 과정은 일반적인 Supervised Learning과 같습니다.

```text
Dataset에서 Batch Sampling
        ↓
Observation 입력
        ↓
Model이 Action 예측
        ↓
Expert Action과 비교
        ↓
Loss 계산
        ↓
Backpropagation
        ↓
Model Update
```

간단한 형태는 다음과 같습니다.

```python
for observation, expert_action in dataloader:

    pred_action = policy(observation)

    loss = criterion(
        pred_action,
        expert_action,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Hybrid Action이라면 acceleration과 lane change loss를 각각 계산하면 됩니다.

---

# 11. 학습 결과 확인

먼저 Training Loss가 감소하는지 확인합니다.

```text
Epoch 1   Loss = 0.82
Epoch 10  Loss = 0.35
Epoch 20  Loss = 0.18
...
```

하지만 Loss가 낮다고 해서 실제 주행을 잘한다는 의미는 아닙니다.

반드시!! 학습된 모델을 다시 SUMO 환경에 넣어 평가해야 합니다.

---

# 12. SUMO에서 BC Policy 평가

학습이 끝나면 AV 차량에 학습된 모델을 불러와서 AV의 Action을 결정하도록 합니다.

```text
SUMO
  ↓
Observation
  ↓
BC Policy
  ↓
Predicted Action
  ↓
AV에 Action 적용
  ↓
다음 Simulation Step
```

즉,

```python
observation = env.get_observation()

with torch.no_grad():
    action = policy(observation)

env.step(action)
```

와 같은 구조입니다.

---

# 13. 무엇을 평가하면 되나요?

프로젝트에서는 복잡한 평가 지표를 많이 사용할 필요는 없습니다.

최소한 다음 정도를 확인하면 됩니다.

### 1. Collision Rate

```text
Collision Rate
=
충돌 Episode 수
/
전체 Evaluation Episode 수
```

프로젝트 기본 목표:

> **Collision Rate ≤ 5%**

---

### 2. Episode Reward

앞에서 정의한 Reward를 이용해 BC Policy가 얼마나 좋은 주행을 하는지 확인할 수 있습니다.

---

### 3. 주행 영상

`sumo-gui`에서 실제로 다음을 확인합니다.

- 차량이 정상적으로 앞으로 진행하는가?
- 앞 차량에 지나치게 가까이 접근하지 않는가?
- 이상한 가속 / 감속을 반복하지 않는가?
- 차선 변경이 정상적으로 이루어지는가?
- 목적한 Route를 정상적으로 통과하는가?

---

# 14. BC의 한계

BC는 Dataset에서 본 행동을 따라 하는 방법입니다.

따라서 학습 중에는 본 적이 없는 상황이 발생하면 성능이 크게 떨어질 수 있습니다.

예를 들어,

```text
IDM Dataset
     ↓
안전한 상태만 많이 포함
     ↓
BC Policy가 작은 실수
     ↓
Dataset에서 거의 보지 못한 State로 이동
     ↓
추가 실수
```

가 발생할 수 있습니다.

이를 **Distribution Shift** 문제라고 합니다.

> **BC는 Dataset의 품질과 다양성에 크게 영향을 받는다**

정도로 이해하면 충분합니다.

---

# 15. Dataset을 잘 모으는 것이 중요합니다

BC에서는 모델 구조보다 Dataset이 중요합니다.

좋은 Dataset을 만들기 위해 다음을 확인하세요.

- [ ] 다양한 Route가 포함되어 있는가?
- [ ] 다양한 차량 밀도가 포함되어 있는가?
- [ ] 가속 / 감속 상황이 모두 포함되어 있는가?
- [ ] Lane Change를 학습한다면 실제 Lane Change 데이터가 충분한가?
- [ ] 비정상적인 Simulation 데이터가 포함되어 있지 않은가?

특히 대부분의 데이터가

```text
Lane Change = Keep
```

뿐이라면 모델도 거의 항상 `Keep`만 출력할 수 있습니다.

---

# Next

다음 자료에서는 Dataset을 따라 하는 것이 아니라,

**Reward를 이용하여 직접 Policy를 학습하는 Reinforcement Learning**을 다룹니다.

### 4. Model Training — Reinforcement Learning

```text
Observation
    ↓
Policy
    ↓
Action
    ↓
SUMO
    ↓
Reward
    ↓
Policy Update
```
