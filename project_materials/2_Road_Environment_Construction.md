# 2. Road Environment Construction

> **목표:** 직접 구축한 SUMO 도로 위에 일반 차량(HV)을 배치하고,  
> 자율주행 차량(AV)이 주변 환경을 관측하고 행동할 수 있도록 **POMDP 구조를 설계**합니다.

---

# 1. 이번 단계에서 무엇을 하나요?

앞 단계에서 프로젝트의 전체 방향과 구현할 도로를 정했다면, 이번 단계에서는 실제로 **차량이 움직일 수 있는 자율주행 환경**을 구성합니다.

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

이번 단계에서 가장 중요한 것은 아직 **좋은 RL 알고리즘을 만드는 것**이 아닙니다.

먼저,

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
road_environment/
│
├── road.net.xml          # 도로 Network
├── traffic.rou.xml       # Vehicle / Route / Flow
├── simulation.sumocfg    # SUMO 실행 설정
└── environment.py        # TraCI를 이용한 Python 환경
```

파일 이름은 자유롭게 변경해도 됩니다.

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

> **도로 자체에 문제가 있으면 이후의 RL 학습으로 해결할 수 없습니다.**

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

SUMO에서는 `.rou.xml` 파일에서 차량의 Route와 Vehicle Type을 정의할 수 있습니다.

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
> 실제 프로젝트에서는 자신이 만든 도로의 길이, 제한속도, 교통량에 맞게 조절해야 합니다.

---

# 5. 일반 차량(HV)을 먼저 만들어야 하는 이유

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

따라서 프로젝트에서는 자율주행 차량 주변에 여러 대의 **일반 차량(Human-driven Vehicle, HV)** 을 배치합니다.

일반 차량들이 만들어내는 교통 흐름 속에서 AV가 적절한 행동을 선택하도록 만드는 것이 목표입니다.

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

# 7. IDM의 기본 아이디어

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

SUMO에는 IDM이 이미 구현되어 있기 때문에 프로젝트에서는

```xml
carFollowModel="IDM"
```

과 같이 Vehicle Type을 설정하여 사용할 수 있습니다.

---

# 8. IDM에서 자주 조절하는 값

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

# 9. 중요: IDM은 차선 변경 모델이 아닙니다

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

# 10. 일반 차량 환경 확인

일반 차량을 추가한 후에는 **AV를 넣기 전에** 교통 흐름부터 확인해야 합니다.

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

# 11. 이제 자율주행 차량(AV)을 추가합니다

일반 차량 환경이 정상적으로 동작하면 일반 차량 중 일부를 **Autonomous Vehicle (AV)** 로 설정합니다.

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

# 12. POMDP란?

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

# 13. State와 Observation의 차이

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

# 14. POMDP의 기본 구조

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

# 15. Observation 설계

프로젝트에서 가장 먼저 결정해야 하는 것은

> **"우리 AV에게 어떤 정보를 보여줄 것인가?"**

입니다.

## 15.1 Ego Vehicle 정보

예:

```text
Ego speed
Ego acceleration
Current lane
Current road position
Distance to goal
```

## 15.2 주변 차량 정보

예:

```text
Front vehicle distance
Front vehicle relative speed

Rear vehicle distance
Rear vehicle relative speed

Left-front vehicle distance
Left-front vehicle relative speed

Left-rear vehicle distance
Left-rear vehicle relative speed

Right-front vehicle distance
Right-front vehicle relative speed

Right-rear vehicle distance
Right-rear vehicle relative speed
```

## 15.3 예시 Observation Vector

예를 들어 다음과 같은 Observation을 만들 수 있습니다.

$$
o_t =
[
v_{\mathrm{ego}},
a_{\mathrm{ego}},
d_{\mathrm{front}},
\Delta v_{\mathrm{front}},
d_{\mathrm{rear}},
\Delta v_{\mathrm{rear}},
l_{\mathrm{ego}}
]
$$

Python에서는 개념적으로 다음과 같은 형태가 됩니다.

```python
observation = [
    ego_speed,
    ego_acceleration,
    front_distance,
    front_relative_speed,
    rear_distance,
    rear_relative_speed,
    ego_lane,
]
```

---

# 16. 주변 차량이 없으면 어떻게 하나요?

예를 들어 관측 범위 안에 앞 차량이 없을 수 있습니다.

그런 경우 Observation의 길이가 계속 바뀌면 Neural Network에 입력하기 어렵습니다.

따라서 Observation의 크기를 **항상 동일하게 유지**해야 합니다.

예:

```text
앞 차량 존재
front_distance = 12.4

앞 차량 없음
front_distance = observation_range
```

또는 별도의 Mask를 사용할 수도 있습니다.

```text
front_exists = 0 or 1
```

중요한 것은

> **모든 timestep에서 Observation의 차원이 동일해야 한다는 것**

입니다.

---

# 17. Observation Normalization

Neural Network 학습에서는 입력 값의 크기가 지나치게 다르면 학습이 어려워질 수 있습니다.

예를 들어,

```text
Speed        = 13.2
Distance     = 87.5
Lane index   = 2
```

와 같이 값의 범위가 서로 다를 수 있습니다.

따라서 필요하다면 다음과 같이 정규화할 수 있습니다.

```python
normalized_speed = speed / max_speed
normalized_distance = distance / observation_range
```

---

# 18. Action 설계

다음으로 결정해야 하는 것은

> **"AV가 어떤 행동을 직접 결정하게 할 것인가?"**

입니다.

## Option A. Discrete Action

가장 단순한 방법입니다.

```text
Action 0 = Decelerate
Action 1 = Keep
Action 2 = Accelerate
```

차선 변경까지 포함한다면,

```text
Action 0 = Decelerate
Action 1 = Keep
Action 2 = Accelerate
Action 3 = Lane Change Left
Action 4 = Lane Change Right
```

처럼 구성할 수 있습니다.

## Option B. Continuous Action

가속도를 연속적인 값으로 직접 출력할 수도 있습니다.

예:

$$
a_t \in [-3.0,\;2.0] \; m/s^2
$$

```text
-3.0  → 강한 감속
-1.0  → 약한 감속
 0.0  → 현재 상태 유지
+1.0  → 가속
+2.0  → 강한 가속
```

## Option C. Hybrid Action

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
a_acc ∈ [-3.0, 2.0]

Lane Change
a_lc ∈ {-1, 0, +1}

-1 = Left
 0 = Keep
+1 = Right
```

처음 프로젝트를 구현할 때는 Action Space를 지나치게 복잡하게 만들 필요가 없습니다.

> **먼저 단순한 Action으로 전체 환경이 정상적으로 동작하는지 확인하고, 이후 확장하는 것을 권장합니다.**

---

# 19. TraCI를 이용한 AV 제어

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

# 20. Reward란?

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

# 21. Reward 설계 예시

가장 단순한 예를 생각해보겠습니다.

$$
r_t
=
w_p r_{\mathrm{progress}}
+
w_v r_{\mathrm{speed}}
-
w_c r_{\mathrm{collision}}
+
w_g r_{\mathrm{goal}}
$$

각 항의 의미는 다음과 같습니다.

| Reward | 의미 |
|---|---|
| $r_{\mathrm{progress}}$ | 목적지를 향해 진행한 정도 |
| $r_{\mathrm{speed}}$ | 적절한 속도로 주행했는지 |
| $r_{\mathrm{collision}}$ | 충돌 여부 |
| $r_{\mathrm{goal}}$ | 목적지 도착 여부 |

예를 들어 코드에서는 다음과 같은 형태가 될 수 있습니다.

```python
reward = 0.0

reward += w_progress * progress_reward
reward += w_speed * speed_reward

if collision:
    reward -= w_collision

if reached_goal:
    reward += w_goal
```

> 위 Reward는 **예시 구조**입니다.  
> 각 팀이 구현한 도로와 AV의 목표에 맞게 수정해야 합니다.

---

# 22. Reward를 너무 복잡하게 만들지 마세요

처음부터 다음과 같이 너무 많은 Reward를 넣으면

```text
speed
+ progress
+ lane keeping
+ comfort
+ jerk
+ time
+ headway
+ lane change
+ traffic efficiency
+ ...
```

차량이 왜 특정 행동을 학습했는지 분석하기 어려워집니다.

처음에는 최소한의 Reward로 시작하는 것을 권장합니다.

```text
Progress
+
Goal
-
Collision
```

환경이 정상적으로 학습되는 것을 확인한 다음 필요한 항목을 하나씩 추가하세요.

---

# 23. Terminal Condition / Done

Episode가 언제 종료되는지도 정의해야 합니다.

예를 들어 다음과 같은 조건을 사용할 수 있습니다.

```text
Collision
→ Episode 종료

Destination 도착
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

# 24. 하나의 Environment Step 정리

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

# 25. Environment 구조 예시

Gym / Gymnasium과 유사한 형태로 환경을 구성한다면 다음과 같은 구조를 사용할 수 있습니다.

```python
class AutonomousDrivingEnv:

    def reset(self):
        # SUMO simulation 초기화
        # 차량 생성
        # Initial observation 반환
        ...

    def get_observation(self):
        # Ego + 주변 차량 정보를 이용하여 Observation 생성
        ...

    def apply_action(self, action):
        # Model이 선택한 Action을 TraCI를 통해 AV에 적용
        ...

    def get_reward(self):
        # 현재 State를 이용하여 Reward 계산
        ...

    def check_done(self):
        # Collision / Goal / Time limit 확인
        ...

    def step(self, action):

        self.apply_action(action)
        traci.simulationStep()

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.check_done()

        return observation, reward, done
```

위 코드는 **구조를 이해하기 위한 예시**이며, 수업에서 사용하는 Gym/Gymnasium 버전에 따라 실제 반환 형태는 달라질 수 있습니다.

---

# 26. HV와 AV의 차이 정리

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

---

# 27. IDM 차량으로 데이터 수집하기

일반 차량이 정상적으로 주행한다면, IDM 차량의 주행 기록을 **Demonstration Dataset**으로 사용할 수도 있습니다.

예를 들어 매 timestep마다 다음 정보를 저장할 수 있습니다.

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

형태의 데이터를 만들 수 있습니다.

## 27.1 BC를 위한 데이터

Behavior Cloning에서는 기본적으로

```text
Observation → Expert Action
```

관계를 학습합니다.

따라서 IDM 차량을 Expert로 사용한다면,

```text
IDM이 어떤 상황에서
어떤 가속 / 감속 행동을 했는가?
```

를 기록할 수 있습니다.

예:

| Step | Ego Speed | Front Distance | Relative Speed | Expert Acceleration |
|---:|---:|---:|---:|---:|
| 0 | 8.2 | 21.4 | -0.5 | 1.10 |
| 1 | 9.1 | 18.8 | 0.3 | 0.72 |
| 2 | 9.7 | 12.1 | 2.1 | -0.85 |

이 데이터를 이용한 실제 BC 학습 방법은 다음 자료에서 다룹니다.

> **3. Model Training — Behavior Cloning**

---

# 28. 데이터 수집 시 가장 중요한 것

데이터에서 사용하는 **Action의 정의와 실제 학습 모델의 Action이 일치해야 합니다.**

예를 들어 AV 모델이

```text
Action = Acceleration
```

을 출력하도록 만들 예정이라면 Dataset에서도

```text
Expert Acceleration
```

을 저장해야 합니다.

즉,

```text
Environment Action Space
        =
Dataset Action
        =
Model Output
```

이 되도록 설계하는 것이 가장 좋습니다.

---

# 29. 권장 구현 순서

## Phase 1 — Road

```text
NetEdit / Network 생성
↓
SUMO-GUI 실행
↓
도로 연결 확인
```

**완료 조건**

> 차량이 없어도 Road Network가 정상적이다.

## Phase 2 — Route

```text
Route 생성
↓
모든 진입 / 진출 방향 확인
```

**완료 조건**

> 각 Route를 따라 차량이 목적지까지 이동할 수 있다.

## Phase 3 — Human Vehicles

```text
IDM Vehicle Type 생성
↓
Flow 생성
↓
Traffic Simulation 실행
```

**완료 조건**

> 일반 차량들이 안정적으로 도로를 주행한다.

## Phase 4 — AV Observation

```text
AV 생성
↓
TraCI 연결
↓
Ego / 주변 차량 정보 읽기
↓
Observation Vector 생성
```

**완료 조건**

> 매 timestep마다 고정된 크기의 Observation을 얻을 수 있다.

## Phase 5 — AV Action

```text
Action 정의
↓
TraCI로 Action 적용
↓
실제 차량 움직임 확인
```

**완료 조건**

> Python에서 내린 Action에 따라 AV가 움직인다.

## Phase 6 — Reward / Done

```text
Reward 계산
↓
Collision 확인
↓
Goal 확인
↓
Episode 종료
```

**완료 조건**

> 한 Episode를 처음부터 끝까지 자동으로 실행할 수 있다.

## Phase 7 — Dataset

```text
Observation
Action
Reward
Next Observation
Done
↓
파일로 저장
```

**완료 조건**

> 여러 Episode의 Trajectory를 저장할 수 있다.

---

# 30. 추천 최소 구현

처음에는 다음 정도로 시작해도 충분합니다.

### Road

```text
실제 한국 도로 기반
+
하나 이상의 의미 있는 Junction
```

### HV

```text
IDM
+
여러 Route
```

### Observation

```text
Ego speed
+
Front distance
+
Front relative speed
```

### Action

```text
Continuous Acceleration
```

### Reward

```text
Progress
+
Goal
-
Collision
```

이 최소 환경이 정상적으로 작동한 이후,

```text
Rear vehicle
Adjacent lane vehicles
Lane change
Complex reward
More traffic
```

등을 추가하면 됩니다.

---

# 31. 자주 발생하는 문제

## Q1. 차량이 Junction에서 계속 멈춰요.

먼저 RL 문제가 아니라 **Road / Connection / Route 문제**인지 확인하세요.

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

---

# 32. 이번 단계의 최종 Checklist

## Road Environment

- [ ] 실제 한국 도로를 참고하여 Road Network를 구축하였다.
- [ ] Edge / Lane / Junction / Connection이 정상적이다.
- [ ] 모든 주요 Route가 정상적으로 연결되어 있다.
- [ ] SUMO-GUI에서 차량이 정상적으로 이동한다.

## Human-driven Vehicles

- [ ] 일반 차량(HV)을 생성하였다.
- [ ] IDM의 역할을 이해하였다.
- [ ] Car-Following과 Lane-Changing의 차이를 이해하였다.
- [ ] 여러 Route에서 차량이 안정적으로 주행한다.

## Autonomous Vehicle

- [ ] AV를 별도로 구분하였다.
- [ ] Observation을 정의하였다.
- [ ] 모든 timestep에서 Observation 크기가 일정하다.
- [ ] Action Space를 정의하였다.
- [ ] TraCI를 통해 Action을 차량에 적용할 수 있다.

## POMDP

- [ ] State와 Observation의 차이를 이해하였다.
- [ ] Reward를 정의하였다.
- [ ] Terminal Condition을 정의하였다.
- [ ] 한 Episode가 자동으로 실행된다.

## Dataset

- [ ] `(observation, action, reward, next_observation, done)`을 저장할 수 있다.
- [ ] Dataset의 Action 정의와 Model의 Action Space가 일치한다.

---

# 33. 이 단계에서 기억해야 할 것

이번 단계의 목표는 **학습 성능을 높이는 것**이 아닙니다.

먼저 다음 Pipeline이 완성되어야 합니다.

```text
Road
 ↓
Human Vehicles
 ↓
Autonomous Vehicle
 ↓
Observation
 ↓
Action
 ↓
Reward
 ↓
Next Observation
```

즉,

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
