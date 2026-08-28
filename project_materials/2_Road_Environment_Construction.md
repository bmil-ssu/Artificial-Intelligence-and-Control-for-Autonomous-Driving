# 2. Road Environment Construction

> **목표:** 직접 구축한 SUMO 도로 위에 일반 차량(HV)을 배치하고,  
> 자율주행 차량(AV)이 주변 환경을 관측하고 행동할 수 있도록 **POMDP 구조를 설계**합니다.

---

# 1. 이번 단계에서 무엇을 하나요?

**차량이 움직일 수 있는 자율주행 환경**을 구성합니다.

전체 흐름은 다음과 같습니다.

```text
SUMO 도로 구조 구축
        ↓
Route / Traffic Flow 정의
        ↓
일반 차량(HV) 추가
        ↓
IDM 기반 차량 주행 확인
        ↓
자율주행 차량(AV) 추가
        ↓
Observation / Action / Reward 정의
        ↓
학습 가능한 Environment 완성
```

> **"내가 만든 도로에서 일반 차량과 자율주행 차량이 정상적으로 움직일 수 있는 환경을 만드는 것"**

이 목표입니다.

---

# 2. SUMO 환경의 기본 구성

SUMO 환경을 구성할 때 자주 보게 되는 요소는 다음과 같습니다.

| 요소 | 의미 |
|---|---|
| **Edge** | 도로 구간 |
| **Lane** | Edge 내부의 차선 |
| **Junction** | 교차로 또는 도로 연결 지점 |
| **Connection** | Lane과 Lane 사이의 연결 관계 |
| **Route** | 차량이 이동할 도로 순서 |
| **Vehicle Type (`vType`)** | 차량의 크기, 최대 속도, 가속도, 주행 모델 등의 설정 |
| **Vehicle / Flow** | 실제로 시뮬레이션에 생성되는 차량 또는 차량 흐름 |

일반적으로 프로젝트 폴더에서는 다음과 같은 파일들을 사용하게 됩니다.

```text
sumo_rl/
│
├─ env/                     ← 환경(문제 정의) 관련 전부
│  ├─ road_config.py        도로/차량/교통류 설정 (파이썬 dict)
│  ├─ mdp_config.py         MDP 정의: 관측·행동·보상·에피소드 길이
│  ├─ road_builder.py       road_config → SUMO 파일 자동 생성 (netconvert)
│  ├─ sumo_env.py           Gymnasium 환경 (TraCI로 SUMO 조종)
│  └─ sumo/                 자동 생성되는 SUMO 파일 (직접 수정하지 말 것)
│
└─ utils/                   ← 알고리즘/환경에 독립적인 재사용 부품
   └─ networks.py           build_mlp, GaussianActorCritic, CategoricalActorCritic
```

---

# 3. 도로 구조 구축

프로젝트에서는 **대한민국에 실제로 존재하는 도로 구조**를 참고하여 SUMO 도로를 만듭니다.

예를 들어,

- 삼거리
- 사거리
- 오거리
- 회전교차로
- 합류 구간
- 분기 구간
- 차로 감소 구간
- 복합 교차로

등을 구현할 수 있습니다.

실제 도로를 완벽하게 1:1로 복사할 필요는 없습니다.

다만 다음과 같은 핵심 구조는 최대한 반영하는 것을 권장합니다.

- 차로 수
- 도로의 진행 방향
- Junction 위치
- 차량 진입 / 진출 방향
- 좌회전 / 우회전 가능 여부
- 차선 간 Connection
- 합류 또는 분기 구조

---

## 3.1 처음에는 도로만 확인하세요

처음부터 차량이나 강화학습 코드를 추가하지 마세요.

먼저 `sumo-gui`에서 도로만 실행하고 아래 항목을 확인합니다.

### Road Checklist

- [ ] 모든 Edge가 정상적으로 연결되어 있는가?
- [ ] Lane의 개수가 의도한 것과 같은가?
- [ ] Junction이 정상적으로 연결되어 있는가?
- [ ] 좌회전 / 직진 / 우회전 Connection이 올바른가?
- [ ] 차량이 진입할 수 없는 끊어진 Lane이 없는가?
- [ ] 도로 방향이 반대로 설정된 곳은 없는가?

> **도로 자체에 문제가 있으면 이후의 모델 학습으로 해결할 수 없습니다.**

따라서 환경 구축 단계에서 충분히 확인해야 합니다.

---

# 4. Route와 Traffic Flow 만들기

도로가 완성되었다면 차량이 **어디에서 출발하고 어디로 이동할지** 정의해야 합니다.

예를 들어 다음과 같은 Route를 생각할 수 있습니다.

```text
북쪽 진입 → 교차로 → 남쪽 진출
서쪽 진입 → 교차로 → 동쪽 진출
남쪽 진입 → 교차로 → 서쪽 진출
```

<!--SUMO에서는 `.rou.xml` 파일에서 차량의 Route와 Vehicle Type을 정의할 수 있습니다.

아래는 개념을 이해하기 위한 간단한 예시입니다.

```xml
<routes>

    <vType
        id="human"
        vClass="passenger"
        carFollowModel="IDM"
        accel="2.6"
        decel="4.5"
        tau="1.2"
        minGap="2.5"
        maxSpeed="13.89"
    />

    <route
        id="route_0"
        edges="edge_0 edge_1 edge_2"
    />

    <flow
        id="human_flow_0"
        type="human"
        route="route_0"
        begin="0"
        end="3600"
        vehsPerHour="600"
    />

</routes>
```

위 예시는 다음을 의미합니다.

```text
human이라는 일반 차량 타입을 만들고
        ↓
IDM을 이용하여 앞 차량을 따라가도록 하고
        ↓
route_0 경로를 따라
        ↓
일정한 교통량으로 차량을 생성
```

> 위의 수치는 **예시**입니다.  
> 실제 프로젝트에서는 자신이 만든 도로의 길이, 제한속도, 교통량에 맞게 조절해야 합니다. -->

---

# 5. 일반 차량(HV)을 먼저 만들어야 하는 이유

도로가 제대로 만들어졌는지 확인하기 위해 HV를 먼저 도로에 배치합니다.

자율주행 차량 한 대만 도로 위에 놓으면 대부분의 의사결정 문제가 너무 단순해집니다.

예를 들어 주변에 다른 차량이 없다면,

```text
앞 차량과 거리 조절
합류
양보
차선 변경
교차로 통과
충돌 회피
```

와 같은 행동이 거의 필요하지 않습니다.

따라서 프로젝트에서는 일반 차량들이 만들어내는 교통 흐름 속에서 AV가 적절한 행동을 선택하도록 만드는 것이 목표입니다.

---

# 6. IDM이란?

## Intelligent Driver Model (IDM)

**IDM(Intelligent Driver Model)** 은 앞 차량과의 거리와 상대 속도를 이용하여 차량의 가속 / 감속을 결정하는 대표적인 **Car-Following Model**입니다.

쉽게 말하면,

> **"앞 차량과 너무 가까우면 감속하고, 충분히 멀면 원하는 속도까지 가속하는 일반 운전자 모델"**

이라고 생각하면 됩니다.

---

## 6.1 IDM이 보는 정보

IDM 차량은 대표적으로 다음 정보를 이용합니다.

```text
현재 차량 속도
+
앞 차량과의 거리
+
앞 차량과의 상대 속도
+
원하는 주행 속도
+
원하는 안전거리
```

### Case 1. 앞 차량이 멀리 있음

```text
🚗 Ego                       🚙 Leader
────────────── 큰 거리 ──────────────

→ 가속
```

### Case 2. 앞 차량과 가까워짐

```text
🚗 Ego       🚙 Leader
──── 짧은 거리 ────

→ 감속
```

### Case 3. 앞 차량이 느려짐

```text
🚗 Ego  →→→     🚙 Leader →

→ 상대 속도 차이가 커짐
→ 더 강하게 감속
```

---

<!-- # 7. IDM의 기본 아이디어

IDM의 대표적인 가속도 모델은 다음과 같이 표현할 수 있습니다.

$$
\dot{v}
=
a
\left[
1-
\left(\frac{v}{v_0}\right)^\delta
-
\left(\frac{s^*(v,\Delta v)}{s}\right)^2
\right]
$$

여기서 원하는 안전거리는 다음과 같이 표현할 수 있습니다.

$$
s^*(v,\Delta v)
=
s_0
+
\max
\left(
0,
vT
+
\frac{v\Delta v}{2\sqrt{ab}}
\right)
$$

각 변수는 대략 다음 의미입니다.

| 기호 | 의미 |
|---|---|
| $v$ | 현재 차량 속도 |
| $v_0$ | 원하는 주행 속도 |
| $s$ | 앞 차량과의 현재 거리 |
| $s_0$ | 최소 안전거리 |
| $T$ | 원하는 Time Headway |
| $\Delta v$ | 앞 차량과의 상대 속도 |
| $a$ | 최대 가속 성향 |
| $b$ | 편안한 감속 성향 |
| $\delta$ | 가속 특성을 결정하는 계수 |

이 수식을 직접 구현하거나 외울 필요는 없습니다.
-->

SUMO에는 IDM이 이미 구현되어 있기 때문에 프로젝트에서는

```xml
carFollowModel="IDM"
```

과 같이 Vehicle Type을 설정하여 사용할 수 있습니다.

---

# 7. IDM에서 자주 조절하는 값

SUMO에서 일반 차량의 성향을 바꾸고 싶다면 다음과 같은 값을 조절할 수 있습니다.

| Parameter | 의미 | 값이 커지면 |
|---|---|---|
| `accel` | 최대 가속 능력 | 더 빠르게 가속 |
| `decel` | 감속 능력 | 더 강한 감속 가능 |
| `tau` | 원하는 시간 간격(Time Headway) | 앞 차량과 더 긴 간격 유지 |
| `minGap` | 최소 차량 간격 | 차량 사이 공간 증가 |
| `maxSpeed` | 차량의 최대 속도 | 더 높은 속도로 주행 가능 |

예:

```xml
<vType
    id="human_cautious"
    carFollowModel="IDM"
    accel="1.8"
    decel="4.0"
    tau="1.5"
    minGap="3.0"
    maxSpeed="13.89"
/>
```

이처럼 차량마다 다른 설정을 주면 조금 더 다양한 교통 흐름을 만들 수 있습니다.

---

# 중요: IDM은 차선 변경 모델이 아닙니다

여기서 한 가지 중요한 점이 있습니다.

> **IDM은 기본적으로 차량의 종방향(Longitudinal) 움직임을 결정하는 Car-Following Model입니다.**

즉,

```text
가속
감속
앞 차량과의 거리 유지
```

를 담당합니다.

반면,

```text
왼쪽 차선으로 이동
오른쪽 차선으로 이동
추월
합류
```

와 같은 **횡방향(Lateral) 의사결정**은 Lane-Changing Model이 담당합니다.

SUMO에서는 예를 들어 `LC2013`과 같은 Lane-Changing Model을 사용할 수 있습니다.

```text
일반 차량(HV)
│
├── Car-Following Model
│       └── IDM
│           └── 가속 / 감속
│
└── Lane-Changing Model
        └── LC2013 등
            └── 차선 변경
```

---

# 8. 일반 차량 환경 확인

일반 차량을 추가한 후에는 교통 흐름부터 확인해야 합니다.

- [ ] 차량이 정상적으로 생성되는가?
- [ ] 차량이 Route를 끝까지 따라가는가?
- [ ] 특정 Junction에서 차량이 비정상적으로 멈추지 않는가?
- [ ] 차량이 계속 Teleport되지 않는가?
- [ ] 비정상적인 충돌이 반복되지 않는가?
- [ ] 교차로에서 Traffic Flow가 자연스러운가?
- [ ] 차량 수가 너무 적거나 지나치게 많지 않은가?

좋은 환경은

```text
도로 구조
+
Route
+
일반 차량
```

만 실행해도 충분히 자연스러운 Traffic Flow가 만들어져야 합니다.

---

# 9. 이제 학습된 자율주행 차량(AV)을 추가합니다

일반 차량 환경이 정상적으로 동작하면 자율주행 차량을 한 대 배치합니다.

```text
                    ┌────────────────────┐
                    │        SUMO        │
                    │                    │
                    │  HV   HV   AV   HV │
                    └─────────┬──────────┘
                              │
                       Observation
                              │
                              ▼
                    ┌────────────────────┐
                    │   AV Controller    │
                    │   BC / RL Model    │
                    └─────────┬──────────┘
                              │
                            Action
                              │
                              ▼
                           SUMO
```

일반 차량은 SUMO의 IDM과 같은 주행 모델이 제어하고, 자율주행 차량은 Python에서 작성한 **Controller / Policy** 가 제어합니다.

Python과 SUMO 사이의 정보 교환에는 주로 **TraCI** 를 사용합니다.

---

# 10. POMDP란?

자율주행 차량의 의사결정 문제는 **POMDP(Partially Observable Markov Decision Process)** 로 생각할 수 있습니다.

왜 **Partially Observable**일까요?

실제 자율주행 차량은 도로 위의 모든 정보를 완벽하게 알 수 없기 때문입니다.

예를 들어 AV가 전체 도로의 모든 차량을 알고 있다고 가정하기보다,

```text
AV 주변 일정 거리
또는
가장 가까운 앞/뒤 차량
```

정도만 관측하도록 만들 수 있습니다.

---

# 11. State와 Observation의 차이

처음 강화학습을 접하면 **State와 Observation**을 혼동하기 쉽습니다.

## State

State는 시뮬레이터 내부의 **전체 환경 상태**라고 생각할 수 있습니다.

예를 들어 SUMO는 실제로 다음 정보를 모두 알고 있습니다.

```text
모든 차량의 위치
모든 차량의 속도
모든 차량의 차선
모든 차량의 Route
Traffic Light 상태
도로 전체의 교통 상황
...
```

이를 전체 State $s_t$라고 생각할 수 있습니다.

## Observation

하지만 자율주행 차량에게 모든 정보를 줄 필요는 없습니다.

AV가 실제로 입력으로 사용하는 정보만을

> **Observation $o_t$**

이라고 합니다.

```text
전체 SUMO State
─────────────────────────────────────
Vehicle 1
Vehicle 2
Vehicle 3
Vehicle 4
...
─────────────────────────────────────
             │
             │ AV 주변 정보만 선택
             ▼
        Observation
────────────────────
Ego speed
Ego lane
Front distance
Front relative speed
Rear distance
Rear relative speed
...
────────────────────
```

즉,

> **SUMO의 전체 상태 중 AV가 실제로 사용할 정보만 선택하여 Observation을 구성**

하게 됩니다.

---

# 12. POMDP의 기본 구조

한 시점 $t$에서 다음과 같이 생각하면 됩니다.

```text
현재 Observation
       o_t
        │
        ▼
┌─────────────────┐
│      Policy     │
└────────┬────────┘
         │
       Action
        a_t
         │
         ▼
┌─────────────────┐
│      SUMO       │
└────────┬────────┘
         │
         ▼
Next Observation + Reward
      o_(t+1), r_t
```

학습에서는 이 과정이 반복됩니다.

```text
o_t
 ↓
a_t
 ↓
SUMO simulation step
 ↓
r_t
 ↓
o_(t+1)
 ↓
...
```

---

# 13. Observation 설계

프로젝트에서 가장 먼저 결정해야 하는 것은

> **"우리 AV에게 어떤 정보를 보여줄 것인가?"**

입니다.

## 13.1 Ego Vehicle 정보

예:

```text
내 현재 위치
내 현재 속도
```

## 13.2 주변 차량 정보

예:

```text
내 주변 차량과 나 사이의 상대 속도
내 주변 차량과 나 사이의 상대 위치
```

## 13.3 도로 환경 정보

예:

```text
내 앞 도로의 차량 밀도
도로가 끊겨 있는지, 연결되어 있는지
```

## 13.4 예시 Observation Vector

예를 들어 다음과 같은 Observation을 만들 수 있습니다.

$$
o_t =
[
v_{\mathrm{ego}},
p_{\mathrm{ego}},
\Delta v,
\Delta p,
density,
lane,
]
$$

---

# 14. 주변 차량이 없으면 어떻게 하나요?

예를 들어 관측 범위 안에 앞 차량이 없을 수 있습니다.

그런 경우 Observation의 길이가 계속 바뀌면 Neural Network에 입력하기 어렵습니다.

따라서 Observation의 크기를 **항상 동일하게 유지**해야 합니다.

중요한 것은

> **모든 timestep에서 Observation의 차원이 동일해야 한다는 것**

입니다.

---

# 15. Observation Normalization

Neural Network 학습에서는 입력 값의 크기가 지나치게 다르면 학습이 어려워질 수 있습니다.

예를 들어,

```text
Speed        = 13.2
Distance     = 87.5
Lane index   = 2
```

와 같이 값의 범위가 서로 다를 수 있습니다.

따라서 다음과 같이 정규화할 수 있습니다.

```python
normalized_speed = speed / max_speed
normalized_distance = distance / observation_range
```

---

# 16. Action 설계

다음으로 결정해야 하는 것은

> **"AV가 어떤 행동을 직접 결정하게 할 것인가?"**

입니다.

## Hybrid Action

가속도와 차선 변경을 동시에 결정할 수도 있습니다.

$$
a_t =
[
a_t^{acc},
a_t^{lc}
]
$$

```text
Acceleration
a_acc ∈ [-1.0, 1.0]

Lane Change
a_lc ∈ {-1, 0, +1}

-1 = Left
 0 = Keep
+1 = Right
```
---

# 17. TraCI를 이용한 AV 제어

Python에서는 TraCI를 통해 SUMO 차량의 상태를 읽거나 차량을 제어할 수 있습니다.

차량 속도 확인:

```python
ego_speed = traci.vehicle.getSpeed(av_id)
```

차선 확인:

```python
lane_id = traci.vehicle.getLaneID(av_id)
```

앞 차량 탐색:

```python
leader = traci.vehicle.getLeader(av_id, dist=30.0)
```

AV의 가속도 적용:

```python
traci.vehicle.setAcceleration(
    av_id,
    acceleration,
    duration=1.0,
)
```

차선 변경:

```python
traci.vehicle.changeLane(
    av_id,
    target_lane,
    duration=1.0,
)
```

## :warning: TraCI 제어 시 주의

TraCI로 속도나 가속도를 명령하더라도 SUMO의 기본 설정에서는 **안전 속도, 최대 가속/감속, 교차로 통행 규칙 등의 제약이 함께 적용될 수 있습니다.**

따라서

```text
Policy가 선택한 Action
```

과

```text
SUMO에서 실제로 적용된 Action
```

이 항상 완전히 같다고 가정하면 안 됩니다.

처음 프로젝트를 구현할 때는 SUMO의 안전 제약을 유지하는 것을 권장합니다.

`speedMode`, `laneChangeMode` 등을 변경할 수도 있지만, 해당 설정의 의미를 이해하지 않은 상태에서 모든 안전 기능을 끄는 것은 권장하지 않습니다.

---

# 18. Reward란?

Reward는 자율주행 차량에게

> **"방금 한 행동이 얼마나 좋은 행동이었는가?"**

를 숫자로 알려주는 값입니다.

예를 들어,

```text
목적지에 도착       → Positive Reward
적절한 속도로 주행 → Positive Reward
앞으로 진행         → Positive Reward

충돌                → Large Negative Reward
도로에서 벗어남     → Negative Reward
지나치게 정지       → Negative Reward
```

처럼 설계할 수 있습니다.

---

## 보상 정의

```
매 스텝:  + speed_weight(0.1) × (내 속도 / vmax)          ← 빠를수록 보상
          - close_gap_penalty(0.2)  (앞차 10m 미만 접근 시) ← 위험운전 감점
종료 시:  충돌 -5 / 완주 +2 / 시간초과(400스텝) 추가보상 없음
```

이상적 에피소드 리턴 ≈ +10, 초반 충돌 시 마이너스권.
값과 설계 이유(스케일을 왜 줄였는지)는 env/mdp_config.py의 REWARD 주석을 참고하세요.

---

# 19. Terminal Condition / Done

Episode가 언제 종료되는지도 정의해야 합니다.

예를 들어 다음과 같은 조건을 사용할 수 있습니다.

```text
Collision
→ Episode 종료

Maximum Simulation Step 도달
→ Episode 종료
```

Python에서는 개념적으로 다음과 같습니다.

```python
done = (
    collision
    or reached_goal
    or step >= max_episode_steps
)
```

---

# 20. 하나의 Environment Step 정리

지금까지의 내용을 하나의 timestep으로 정리하면 다음과 같습니다.

```text
1. SUMO에서 현재 차량 정보 읽기
            ↓
2. Observation o_t 구성
            ↓
3. Controller / Policy가 Action a_t 결정
            ↓
4. Action을 SUMO에 적용
            ↓
5. SUMO simulationStep()
            ↓
6. Reward r_t 계산
            ↓
7. Done 여부 확인
            ↓
8. Next Observation o_(t+1) 생성
```

이를 반복하면 자율주행 학습 환경이 됩니다.

---

# 21. HV와 AV의 차이 정리

| | 일반 차량 (HV) | 자율주행 차량 (AV) |
|---|---|---|
| Controller | SUMO 기본 모델 | 학습 모델 / Python Controller |
| 종방향 제어 | IDM 등 | Policy Action |
| 차선 변경 | SUMO Lane-Changing Model | SUMO 또는 Policy |
| 주변 정보 | SUMO 내부적으로 사용 | 직접 Observation 구성 |
| 학습 대상 | X | O |
| 역할 | Traffic 생성 | 의사결정 학습 |

즉,

```text
HV = 주변 교통 환경을 만드는 차량

AV = 우리가 학습시키려는 차량
```

이라고 생각하면 됩니다.

# 자주 발생하는 문제

## Q1. 차량이 Junction에서 계속 멈춰요.

**Road / Connection / Route 문제**인지 확인하세요.

```text
Edge 연결
Lane Connection
Right-of-way
Route
```

를 먼저 확인합니다.

## Q2. 차량이 너무 많이 막혀요.

다음 값을 확인합니다.

```text
Traffic Flow
Vehicle insertion rate
tau
minGap
Road capacity
```

차량 수가 도로 용량보다 지나치게 많을 수도 있습니다.

## Q3. AV가 Action대로 움직이지 않아요.

TraCI Action에 대해 SUMO의

```text
Car-Following safety
speedMode
laneChangeMode
```

등이 영향을 주고 있는지 확인합니다.

## Q4. Observation 크기가 계속 달라져요.

주변 차량이 없을 때 사용할 **default value 또는 mask**를 정의하여 입력 차원을 고정합니다.

## Q5. 학습이 잘 안 돼요.

이 단계에서는 먼저 RL 알고리즘을 의심하지 마세요.

```text
1. Road가 정상인가?
2. Route가 정상인가?
3. HV가 정상적으로 움직이는가?
4. Observation 값이 올바른가?
5. Action이 실제 차량에 적용되는가?
6. Reward가 예상대로 계산되는가?
7. Done 조건이 정상인가?
8. 그 다음 학습 알고리즘을 확인한다.
```

> **강화학습을 하기 전에 강화학습이 가능한 환경부터 완성해야 합니다.**

환경이 정상적으로 만들어졌다면 이후에는 같은 Environment를 기반으로

- **Behavior Cloning**
- **Reinforcement Learning**

을 적용할 수 있습니다.

---

# Next

다음 자료에서는 구축한 환경과 수집한 데이터를 이용하여 모델을 학습합니다.

### 3. Model Training — Behavior Cloning

```text
IDM / Expert Dataset
        ↓
Observation → Action
        ↓
Behavior Cloning
```

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
```

---

# References

- SUMO Documentation — Vehicle Types and Routes  
  https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html

- SUMO Documentation — Car-Following Models  
  https://sumo.dlr.de/userdoc/Car-Following-Models/

- SUMO Documentation — TraCI Vehicle State Control  
  https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html
