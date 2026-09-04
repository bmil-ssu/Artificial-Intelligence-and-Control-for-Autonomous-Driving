# 2. Road Environment Construction

> **목표:** 직접 구축한 SUMO 도로 위에 일반 차량(HV)을 배치하고,  
> 자율주행 차량(AV)이 주변 환경을 관측하고 행동할 수 있도록 **학습 가능한 주행 Environment** 구성

---

# 1. 코드

**도로를 만들고, 교통 흐름을 생성하고, AV가 학습할 수 있는 환경을 구성하는 코드** 확인

```text
Artificial-Intelligence-and-Control-for-Autonomous-Driving/
│
├── view_road.py
└── env/
    ├── road_config.py
    ├── road_builder.py
    ├── mdp_config.py
    ├── sumo_env.py
    └── sumo/
        ├── *.net.xml
        ├── *.rou.xml
        └── *.sumocfg
```

| 파일 | 역할 | 주로 수정하는 내용 |
|---|---|---|
| `view_road.py` | 현재 설정된 도로와 교통 환경을 SUMO GUI에서 확인 | 일반적으로 직접 수정하지 않음 |
| `env/road_config.py` | 도로 구조, 차량 종류, 교통량 등 **물리적인 도로 환경 설정** | 차선, 도로 길이, 제한속도, HV 구성, 교통량 |
| `env/road_builder.py` | `road_config.py`를 읽어 SUMO 실행 파일을 **자동 생성** | 일반적으로 직접 수정하지 않음 |
| `env/mdp_config.py` | AV의 **Observation, Action, Reward, Episode, GUI 설정** | 관측 범위, 행동 범위, Reward, Episode 길이 |
| `env/sumo_env.py` | Python과 SUMO를 연결하는 **Gymnasium Environment** | Observation 생성, Action 적용, Reward 및 종료 조건 |
| `env/sumo/` | `road_builder.py`가 자동 생성한 SUMO 파일 저장 | 직접 수정하지 않는 것을 권장 |

## 1.1 코드가 연결되는 순서

```text
road_config.py
     │  도로 / 차량 / 교통량 설정
     ▼
road_builder.py
     │  SUMO 파일 자동 생성
     ▼
env/sumo/
     │  Network / Route / SUMO Configuration
     ▼
sumo_env.py
     │  TraCI를 이용하여 Python ↔ SUMO 연결
     ▼
Observation → Action → Reward
```

현재 도로가 의도한 대로 만들어졌는지 다음 명령으로 확인 가능함

```bash
python view_road.py
```

## 1.2 값 바꿀 때 확인할 파일

| 바꾸고 싶은 것 | 확인할 파일 |
|---|---|
| 도로 길이 / 차선 수 / 제한속도 | `env/road_config.py` |
| 일반 차량 수 / Traffic Flow | `env/road_config.py` |
| HV Controller와 주행 특성 | `env/road_config.py` |
| SUMO Network/Route 생성 방식 | `env/road_builder.py` |
| 도로를 GUI로 확인 | `view_road.py` |
| AV 관측 범위 | `env/mdp_config.py` |
| Observation 구성 | `env/mdp_config.py`, `env/sumo_env.py` |
| Action 범위 | `env/mdp_config.py`, `env/sumo_env.py` |
| Reward 값 / Episode 길이 | `env/mdp_config.py` |
| 실제 Reward 계산 / Done 판단 | `env/sumo_env.py` |
| SUMO GUI 확대 / Delay | `env/mdp_config.py` |

---

# 2. 환경 구성

**차량이 움직일 수 있는 자율주행 환경** 구성

```text
SUMO 도로 구조 구축
        ↓
Route / Traffic Flow 정의
        ↓
일반 차량(HV) 추가
        ↓
교통 흐름 확인
        ↓
자율주행 차량(AV) 추가
        ↓
Observation / Action / Reward 정의
        ↓
학습 가능한 Environment 완성
```

> **핵심 목표:** 내가 만든 도로에서 HV와 AV가 정상적으로 움직이고, AV가 Observation을 받아 Action을 수행할 수 있는 환경을 만드는 것.

---

# 3. `road_config.py` — 도로와 교통 환경 설정

`env/road_config.py`는 **어떤 도로에서 어떤 차량들이 주행할 것인지** 정의

```text
도로 구조 / 차선 수 / 도로 길이 / 제한속도
일반 차량(HV)의 수와 Traffic Flow
차량 Controller와 주행 특성
```

즉, 물리적인 도로나 주변 Traffic을 바꾸고 싶다면 가장 먼저 `road_config.py`를 확인하면 됨

## 3.1 SUMO 도로의 기본 요소

| 요소 | 의미 |
|---|---|
| **Edge** | 도로 구간 |
| **Lane** | Edge 내부의 차선 |
| **Junction** | 도로 연결 지점 |
| **Connection** | Lane과 Lane 사이의 연결 관계 |
| **Route** | 차량이 이동하는 도로 순서 |
| **Vehicle Type (`vType`)** | 차량 크기, 최대 속도, 가속도, 주행 모델 등의 설정 |
| **Vehicle / Flow** | 실제 시뮬레이션에 생성되는 차량 또는 차량 흐름 |

도로를 설계할 때는 **차로 수, 진행 방향, Junction, 진입/진출 방향, Lane Connection, 합류/분기 구조**를 우선 확인

## 3.2 도로를 수정할 때 작업 흐름

```text
road_config.py 수정
        ↓
road_builder.py가 SUMO 파일 생성
        ↓
python view_road.py
        ↓
SUMO GUI에서 확인
```

`env/sumo/` 내부의 XML 파일은 자동으로 다시 생성될 수 있으므로, 직접 수정하기보다 `road_config.py`를 수정하는 것을 권장

---

# 4. `road_builder.py` — SUMO 파일 자동 생성

`env/road_builder.py`는 `road_config.py`의 Python 설정을 SUMO가 읽을 수 있는 Network, Route, Vehicle Type, Traffic Flow, SUMO Configuration 파일로 변환

```text
road_config.py
       ↓
road_builder.py
       ↓
SUMO XML / Configuration files
```

일반적인 실습에서는 `road_builder.py` 자체보다 `road_config.py`의 설정값 수정

> **주의:** `env/sumo/`에 직접 수정한 내용은 다음 생성 시 덮어쓰여질 수 있음

---

# 5. `view_road.py` — 학습 전에 환경 확인

도로와 Traffic Flow를 만든 다음 환경부터 확인

```bash
python view_road.py
```

SUMO GUI가 열리면 **▶ Play** 버튼을 눌러 차량 움직임 확인

### Road Checklist

- [ ] Edge와 Lane이 정상적으로 연결되어 있는가?
- [ ] 차선 수와 진행 방향이 의도한 것과 같은가?
- [ ] 합류 / 분기 Connection이 정상적인가?
- [ ] 차량이 이동할 수 없는 끊어진 구간이 없는가?

### Traffic Checklist

- [ ] 차량이 정상적으로 생성되는가?
- [ ] 차량이 Route를 따라 끝까지 이동하는가?
- [ ] 특정 구간에서 비정상적으로 멈추지 않는가?
- [ ] Teleport나 비정상적인 충돌이 반복되지 않는가?
- [ ] 교통량이 지나치게 많거나 적지 않은가?

> **도로 자체에 문제가 있으면 강화학습으로 해결 불가함**

## 5.1 GUI 확대 정도 조절

GUI 관련 설정은 `env/mdp_config.py`에서 확인

```python
SIMULATION = {
    ...
    "gui_zoom": 4000.0,
    "gui_view_width": 500.0,
    "gui_track_ego": True,
}
```

처음 GUI가 열렸을 때 도로를 **더 확대**해서 보고 싶다면 `gui_view_width`를 줄임

```python
"gui_view_width": 300.0,
```

주행 중 Ego 차량 주변을 더 확대하려면 `gui_zoom`을 높임

```python
"gui_zoom": 5000.0,
```

---

# 6. 일반 차량(HV)과 Traffic 구성

자율주행 차량 한 대만 존재하면 앞차와 거리 조절, 합류, 양보, 차선 변경, 충돌 회피와 같은 의사결정 문제가 충분히 발생하지 않게 됨

따라서 주변 일반 차량(HV)이 **학습에 필요한 Traffic Environment**를 만듦

하나의 Car-Following Model만 사용하는 대신 여러 Controller를 혼합할 수 있음. 실제 비율과 파라미터는 `env/road_config.py`의 Traffic/Controller 관련 설정 확인

| Controller | 특징 |
|---|---|
| **Krauss** | SUMO의 대표적인 기본 차량 추종 모델 |
| **IDM** | 거리와 상대 속도를 이용한 부드러운 차량 추종 |
| **EIDM** | IDM을 확장한 차량 추종 모델 |
| **ACC** | 일정한 차간시간을 유지하는 차량 제어 모델 |

```text
road_config.py
       ↓
HV 종류 + 비율 + 주행 특성
       ↓
road_builder.py
       ↓
SUMO Vehicle Type / Flow
```

## 6.1 IDM이란?

**IDM(Intelligent Driver Model)** 은 앞 차량과의 거리와 상대 속도를 이용하여 가속/감속을 결정하는 대표적인 **Car-Following Model**

> 앞 차량이 가까우면 감속하고, 충분히 멀면 원하는 속도까지 가속하는 모델

SUMO에는 IDM이 구현되어 있으므로 직접 수식을 구현할 필요 없음

```xml
carFollowModel="IDM"
```

### 자주 조절하는 차량 파라미터

| Parameter | 의미 | 값이 커지면 |
|---|---|---|
| `accel` | 최대 가속 능력 | 더 빠르게 가속 가능 |
| `decel` | 감속 능력 | 더 강한 감속 가능 |
| `tau` | 원하는 Time Headway | 앞 차량과 더 긴 간격 유지 |
| `minGap` | 최소 차량 간격 | 최소 공간 증가 |
| `maxSpeed` | 최대 속도 | 더 높은 속도로 주행 가능 |

### IDM과 차선 변경은 다름!

IDM은 종방향(Longitudinal) 가속/감속을 담당, 차선 변경은 SUMO의 Lane-Changing Model 또는 AV Policy가 담당

```text
일반 차량(HV)
│
├── Longitudinal Control
│       └── IDM / Krauss / EIDM / ACC
│
└── Lateral Control
        └── SUMO Lane-Changing Model
```

---

# 7. `mdp_config.py` — AV의 문제 정의

도로와 HV가 준비되었다면 이제 AV에게 **무엇을 보여주고, 어떤 행동을 하게 하며, 어떻게 평가할지** 정의

```text
Observation
Action
Reward
Simulation / Episode length
GUI setting
```

## 7.1 `road_config.py`와 `mdp_config.py`의 차이

```text
road_config.py
→ 어떤 도로와 Traffic Environment인가?

mdp_config.py
→ 그 환경에서 AV가 어떤 문제를 풀 것인가?
```

---

# 8. Observation — AV가 무엇을 보는가?

SUMO는 모든 차량의 상태를 알고 있지만, AV Policy에는 필요한 정보만 전달하며, 이 입력을 **Observation**이라고 함

```text
SUMO 전체 상태
      ↓
AV 주변의 필요한 정보만 선택
      ↓
Observation
```

```text
Ego speed / position / lane
주변 차량과의 상대 위치
주변 차량과의 상대 속도
```

### Observation 관련 코드

```text
mdp_config.py
→ 관측 범위와 Observation 관련 설정

sumo_env.py
→ TraCI에서 값을 읽고 실제 Observation Vector 생성
```

Observation의 dimension은 모든 timestep에서 동일해야 함. 주변 차량이 없을 때는 padding, default value, mask 등을 사용해 입력 크기 고정

필요한 경우 속도와 거리처럼 범위가 다른 입력 정규화

```python
normalized_speed = speed / max_speed
normalized_distance = distance / observation_range
```

---

# 9. Action — AV가 무엇을 결정하는가?

AV Policy는 예를 들어 **가속도 + 차선 변경**의 Hybrid Action을 출력할 수 있음

\[
a_t = [a_t^{acc}, a_t^{lc}]
\]

```text
Acceleration
+
Lane Change {-1, 0, +1}
```

Action의 범위는 `mdp_config.py`에서 확인하고, 실제 SUMO 차량에 적용하는 로직은 `sumo_env.py`에서 확인

---

# 10. Reward와 Episode 종료

Reward는 AV가 수행한 행동을 평가하는 값

```text
빠른 진행 / 목적지 도착      → Positive Reward
앞차에 과도하게 접근         → Penalty
충돌                         → Large Negative Reward
```

Reward 관련 **설정값**은 `mdp_config.py`, 실제 **계산 로직**은 `sumo_env.py`에서 확인

Episode는 일반적으로 다음 상황에서 종료하게 됨

```text
Collision
Goal Reached
Maximum Episode Step
```

Gymnasium에서는 보통 환경 자체 종료(`terminated`)와 시간 제한 종료(`truncated`)를 구분

---

# 11. `sumo_env.py` — 실제 강화학습 Environment

`env/sumo_env.py`는 Python과 SUMO를 연결하는 핵심 코드

```text
SUMO 실행 / TraCI 연결
AV 상태 읽기
Observation 생성
Action 적용
simulationStep 수행
Reward 계산
Done 판단
```

## 11.1 `reset()`

새 Episode를 시작하고 초기 Observation을 반환

```text
새 SUMO Simulation 시작
        ↓
차량 초기화 / Warm-up
        ↓
초기 Observation 생성
```

강화학습 코드에서는 보통 다음과 같이 사용

```python
obs, info = env.reset()
```

## 11.2 `step(action)`

Policy가 결정한 Action을 Environment에 전달

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

하나의 step에서는 대략 다음 과정이 수행됨

```text
1. Action 입력
       ↓
2. AV에 Action 적용
       ↓
3. SUMO simulationStep()
       ↓
4. 충돌 / 도착 여부 확인
       ↓
5. Reward 계산
       ↓
6. Next Observation 생성
       ↓
7. 결과 반환
```

---

# 12. TraCI — Python에서 SUMO 제어

TraCI는 Python에서 SUMO 차량 상태를 읽고 차량을 제어하는 인터페이스

실제 프로젝트에서는 이러한 명령이 `sumo_env.py` 내부에서 사용

### 차량 속도

```python
ego_speed = traci.vehicle.getSpeed(av_id)
```

### 차선 확인

```python
lane_id = traci.vehicle.getLaneID(av_id)
```

### 앞 차량 탐색

```python
leader = traci.vehicle.getLeader(av_id, dist=30.0)
```

### 가속도 적용

```python
traci.vehicle.setAcceleration(av_id, acceleration, duration=1.0)
```

### 차선 변경

```python
traci.vehicle.changeLane(av_id, target_lane, duration=1.0)
```

> **주의:** Policy가 선택한 Action과 SUMO에서 실제 적용된 차량 움직임이 완전히 같지 않을 수 있음, `speedMode`, `laneChangeMode`, Car-Following safety, 최대 가감속 등의 SUMO 제약이 함께 작동할 수 있음

---

# 13. 하나의 Environment Step 전체 흐름


```text
1. SUMO에서 현재 차량 상태 읽기
            ↓
2. Observation o_t 생성
            ↓
3. Policy가 Action a_t 결정
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

이를 반복하면 강화학습 가능한 Environment가 됨

---

# 14. HV와 AV의 차이

| | 일반 차량 (HV) | 자율주행 차량 (AV) |
|---|---|---|
| Controller | SUMO 차량 모델 | Python Policy |
| 종방향 제어 | IDM / Krauss / EIDM / ACC 등 | Policy Action |
| 차선 변경 | SUMO Lane-Changing Model | Policy 또는 SUMO 설정 |
| 주변 정보 | SUMO 내부적으로 사용 | 직접 Observation 구성 |
| 학습 대상 | X | O |
| 역할 | Traffic 생성 | 의사결정 학습 |

```text
HV = 주변 교통 상황을 만들어주는 차량
AV = 우리가 학습시키는 차량
```

---

# 15. 자주 발생하는 문제

## Q1. 차량이 Junction에서 계속 멈추는 경우

**Road / Connection / Route 문제**인지 확인

```text
Edge 연결
Lane Connection
Right-of-way
Route
```

를 먼저 확인

## Q2. 차량이 너무 많이 막히는 문제

다음 값 확인

```text
Traffic Flow
Vehicle insertion rate
tau
minGap
Road capacity
```

차량 수가 도로 용량보다 지나치게 많을 수도 있음

## Q3. AV가 Action대로 움직이지 않는 문제

TraCI Action에 대해 SUMO의

```text
Car-Following safety
speedMode
laneChangeMode
```

등이 영향을 주고 있는지 확인

## Q4. Observation 크기가 계속 달라지는 문제

주변 차량이 없을 때 사용할 **default value 또는 mask**를 정의하여 입력 차원을 고정

## Q5. 학습이 잘 안 되는 문제

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

> **강화학습을 하기 전에 강화학습이 가능한 환경부터 완성해야 함.**

환경이 정상적으로 만들어졌다면 이후에는 같은 Environment를 기반으로

- **Imitation Learning**
- **Reinforcement Learning**

을 적용할 수 있음

---


# Next

다음 시간에는 구축한 환경과 수집한 데이터를 이용하여 모델 학습

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
