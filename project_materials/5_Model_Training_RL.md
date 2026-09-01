# 5. Model Training — Reinforcement Learning

> **목표:** SUMO 환경과 직접 상호작용하면서 Reward를 이용해  
> 자율주행 차량(AV)의 의사결정 Policy 학습

---

# 1. 목표

지난 수업에서는 IDM 차량의 주행 데이터를 이용하여 **Behavior Cloning (BC)** 모델을 학습했습니다.

이번에는

> **AV가 SUMO 환경에서 직접 행동하고, 그 결과로 받은 Reward를 이용해 Policy를 학습합니다.**

전체 흐름은 다음과 같습니다.

```text
Observation
    ↓
Policy
    ↓
Action
    ↓
SUMO
    ↓
Reward + Next Observation
    ↓
Policy Update
    ↓
반복
```

---

# 2. Reinforcement Learning이란?

**Reinforcement Learning (RL)** 은 Agent가 Environment와 반복적으로 상호작용하면서  
높은 Reward를 받을 수 있는 행동을 학습하는 방법입니다.

| RL 구성 요소 | 프로젝트에서의 의미 |
|---|---|
| **Agent** | 자율주행 차량 (AV) |
| **Environment** | SUMO 도로 환경 |
| **Observation** | AV가 관측하는 주변 차량 및 도로 정보 |
| **Action** | Acceleration / Lane Change |
| **Reward** | 안전하고 효율적인 주행에 대한 점수 |
| **Policy** | Observation을 보고 Action을 결정하는 모델 |

즉,

```text
AV가 현재 상황을 보고
        ↓
행동을 선택하고
        ↓
SUMO에서 실제로 움직인 뒤
        ↓
그 결과에 대한 Reward를 받으며
        ↓
더 좋은 행동을 선택하도록 학습
```

하는 과정입니다.

---

# 3. Behavior Cloning과 무엇이 다른가요?

Behavior Cloning은 Dataset에 있는 Action을 따라하도록 학습합니다.

```text
Observation
    ↓
BC Policy
    ↓
Expert Action과 비교
    ↓
Supervised Learning
```

반면 Reinforcement Learning은 정답 Action이 주어지지 않습니다.

```text
Observation
    ↓
RL Policy
    ↓
Action
    ↓
SUMO
    ↓
Reward
    ↓
Policy Update
```

정리하면 다음과 같습니다.

| | Behavior Cloning | Reinforcement Learning |
|---|---|---|
| 학습 기준 | Expert Action | Reward |
| Environment interaction | 학습 중 필요 없음 | 필요 |
| Dataset | 미리 수집 | 학습 중 생성 가능 |
| 목표 | Expert 행동 모방 | 높은 누적 Reward 획득 |

---

# 4. RL에서 저장하는 데이터

한 번의 Environment Step에서 다음과 같은 Transition이 만들어집니다.

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

형태입니다.

예:

| Step | Observation | Action | Reward | Next Observation | Done |
|---:|---|---|---:|---|---|
| 0 | `[0.41, 0.72, ...]` | `[0.31, 0]` | 0.14 | `[0.43, 0.68, ...]` | False |
| 1 | `[0.43, 0.68, ...]` | `[0.12, 0]` | 0.17 | `[0.46, 0.61, ...]` | False |
| 2 | `[0.46, 0.61, ...]` | `[-0.25, -1]` | -0.30 | `[0.44, 0.55, ...]` | False |

---

# 5. 한 번의 RL Step

한 timestep에서 수행되는 과정은 다음과 같습니다.

```text
1. Observation o_t 생성
        ↓
2. Policy가 Action a_t 선택
        ↓
3. Action을 SUMO에 적용
        ↓
4. simulationStep()
        ↓
5. Reward r_t 계산
        ↓
6. Next Observation o_(t+1) 생성
        ↓
7. Done 여부 확인
```

이 과정이 Episode가 끝날 때까지 반복됩니다.

---

# 6. Episode란?

**Episode**는 하나의 주행 시작부터 종료까지의 과정입니다.

예를 들어 다음 상황에서 Episode를 종료할 수 있습니다.

```text
Collision
→ Episode 종료

Maximum Simulation Step 도달
→ Episode 종료

목적지 도착
→ Episode 종료
```

하나의 Episode에서는 다음과 같은 trajectory가 만들어집니다.

```text
o_0
 ↓
a_0
 ↓
r_0
 ↓
o_1
 ↓
a_1
 ↓
r_1
 ↓
...
```

---

# 7. Reward가 잘못 설계되면 어떻게 되나요?

RL Agent는 우리가 의도한 행동이 아니라  
**Reward를 가장 많이 받는 행동**을 학습합니다.

예를 들어 속도 Reward만 너무 크게 주면

```text
높은 속도
      ↓
높은 Reward
      ↓
위험한 주행
      ↓
충돌 증가
```

가 발생할 수 있습니다.

반대로 Collision Penalty가 지나치게 크고 진행 Reward가 너무 작다면

```text
움직이지 않음
      ↓
충돌하지 않음
      ↓
상대적으로 높은 Return
```

처럼 차량이 거의 움직이지 않는 Policy를 학습할 수도 있습니다.

따라서 학습 결과가 이상하다면 **알고리즘뿐 아니라 Reward도 확인**해야 합니다.

---

# 8. Policy Model

RL Policy도 BC와 마찬가지로 Observation을 입력으로 받습니다.

```text
Observation
      ↓
     MLP
      ↓
    Action
```

하지만 BC와 달리 Expert Action을 직접 맞추는 것이 아니라  
RL 알고리즘의 학습 규칙에 따라 Policy가 업데이트됩니다.

---

# 9. Action Space에 따른 RL 알고리즘

Action 형태에 따라 사용할 수 있는 RL 알고리즘이 달라질 수 있습니다.

## Hybrid Action

```text
Acceleration + Lane Change
```

를 함께 사용하는 경우에는 Action Space를 적절히 구성해야 합니다.

```text
Acceleration
→ Continuous

Lane Change
→ Discrete
```

---

# 10. Exploration이 필요한 이유

BC는 Expert Action을 그대로 학습하지만, RL은 Agent가 직접 행동을 시도해야 합니다.

초기 Policy는 아직 학습되지 않았기 때문에 다양한 Action을 시도하면서

```text
어떤 행동이 좋은 Reward를 주는지
```

알아내야 합니다.

이를 **Exploration**이라고 합니다.

예를 들어,

```text
현재 Observation
      ↓
Policy Action
      +
Exploration
      ↓
실제 Action
```

과 같은 형태로 학습할 수 있습니다.

초기에는 다양한 행동을 시도하고, 학습이 진행될수록 더 좋은 행동을 선택하도록 만드는 것이 일반적입니다.

---

# 11. Replay Buffer

DDPG, TD3, SAC와 같은 **Off-policy RL**에서는 학습 중 생성된 Transition을 Replay Buffer에 저장할 수 있습니다.

```text
SUMO Interaction
       ↓
(o, a, r, o', done)
       ↓
Replay Buffer
       ↓
Random Batch Sampling
       ↓
Model Update
```

개념적으로 다음과 같습니다.

```python
replay_buffer.add(
    observation,
    action,
    reward,
    next_observation,
    done,
)
```

학습할 때는 저장된 데이터 중 일부를 Batch로 꺼냅니다.

```python
batch = replay_buffer.sample(batch_size)
```

BC에서 Dataset을 미리 만들어 사용했다면,

RL에서는 **주행하면서 Dataset이 계속 만들어진다**고 생각할 수 있습니다.

---

# 12. Online RL Training

전체 학습 과정은 다음과 같이 생각하면 됩니다.

```text
Environment Reset
        ↓
Observation
        ↓
Policy → Action
        ↓
SUMO Step
        ↓
Reward / Next Observation
        ↓
Transition 저장
        ↓
Model Update
        ↓
다음 Step
        ↓
Episode 종료
        ↓
다시 Reset
```

간단한 구조는 다음과 같습니다.

```python
for episode in range(num_episodes):

    observation = env.reset()
    done = False

    while not done:

        action = policy.select_action(observation)

        next_observation, reward, done = env.step(action)

        replay_buffer.add(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

        policy.update(replay_buffer)

        observation = next_observation
```

실제 학습 코드는 사용하는 RL 알고리즘에 따라 달라질 수 있습니다.

---

# 13. Training과 Evaluation을 구분하세요

학습 중에는 Exploration이 포함될 수 있습니다.

하지만 Evaluation에서는 가능한 한 **학습된 Policy 자체의 성능**을 확인해야 합니다.

```text
Training
Policy + Exploration
        ↓
Action
```

```text
Evaluation
Learned Policy
      ↓
Action
```

따라서 Evaluation 시에는

- Exploration noise 제거
- 동일한 Evaluation 조건 사용
- 여러 Episode 반복

등을 권장합니다.

---

# 14. 무엇을 평가하면 되나요?

## 1. Collision Rate

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

## 2. Episode Reward

한 Episode 동안 받은 Reward의 합을 비교할 수 있습니다.

```text
Episode Return
=
r_0 + r_1 + r_2 + ... + r_T
```

학습이 진행되면서 Episode Reward가 증가하는지 확인합니다.

---

## 3. 주행 결과

`sumo-gui`에서 실제로 확인합니다.

- 차량이 정상적으로 진행하는가?
- 충돌이 자주 발생하지 않는가?
- 위험하게 앞 차량에 접근하지 않는가?
- 불필요한 가속 / 감속을 반복하지 않는가?
- 차선 변경이 정상적인가?
- 도로 구조를 정상적으로 통과하는가?

---

# 15. Learning Curve

RL 학습 과정에서는 Reward의 변화를 그래프로 확인하는 것이 좋습니다.

예:

```text
Episode
   ↓
Episode Reward
```

학습이 정상적으로 진행된다면 전체적으로

```text
낮은 Reward
     ↓
학습 진행
     ↓
높은 Reward
```

방향으로 변화하는 것을 기대할 수 있습니다.

다만 RL은 학습 과정의 변동성이 크기 때문에 Reward가 매 Episode마다 일정하게 증가하지는 않습니다.

---

# 16. 학습이 잘 안 될 때 확인할 것

RL 성능이 좋지 않다고 해서 바로 알고리즘을 바꾸지 마세요.

다음 순서로 확인하는 것을 권장합니다.

```text
1. Road / Route가 정상인가?
2. HV가 정상적으로 움직이는가?
3. Observation 값이 올바른가?
4. Observation normalization이 정상인가?
5. Action이 실제 AV에 적용되는가?
6. Reward가 의도대로 계산되는가?
7. Done 조건이 올바른가?
8. 그 다음 RL 학습 설정을 확인
```

특히

```text
Observation
Action
Reward
```

중 하나라도 잘못 정의되어 있으면 RL 알고리즘을 바꿔도 성능이 좋아지지 않습니다.

---

# 17. BC와 RL 결과 비교

BC와 RL을 모두 학습했다면 같은 환경에서 비교할 수 있습니다.

예:

| Model | Collision Rate | Episode Reward |
|---|---:|---:|
| IDM | ... | ... |
| BC | ... | ... |
| RL | ... | ... |

중요한 것은 모든 모델을 가능한 한 **동일한 Evaluation 환경에서 비교**하는 것입니다.

예:

```text
같은 Road
같은 Traffic 설정
같은 Episode 길이
같은 Reward
같은 Evaluation 횟수
```

---

# 18. 프로젝트에서 중요한 것

이번 프로젝트에서 목표는 새로운 RL 알고리즘을 개발하는 것이 아닙니다.

기본 제공 알고리즘을 사용해도 충분합니다.

더 중요한 것은

```text
직접 구축한 Road Environment
+
정상적인 Observation / Action / Reward
+
안정적인 SUMO Interaction
+
학습된 AV의 실제 주행
```

입니다.

추가적인 RL 알고리즘 적용이나 성능 개선은 **Bonus**로 진행할 수 있습니다.
