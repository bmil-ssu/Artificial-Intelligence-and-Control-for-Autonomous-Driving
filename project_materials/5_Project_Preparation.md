# 5. Project Preparation — 프로젝트 진행을 위한 코드 소개

> **목표:** 제공된 기본 코드를 팀 프로젝트의 도로 구조와 주행 문제에 맞게 수정하고,  
> **도로 구축 → 환경 검증 → 모델 학습 → 데이터 수집 → 최종 평가**의 전체 흐름 이해

---

## 1. 프로젝트에서 구현할 내용

이번 프로젝트에서는 제공된 직선 도로 예제를 그대로 사용하는 것이 아니라, 대한민국에 실제로 존재하는 도로의 특징을 반영한 새로운 SUMO 환경을 구축함.

최종적으로 다음 과정이 하나의 코드 흐름으로 연결되어야 함.

```text
실제 도로 선정
        ↓
SUMO 도로와 Route 구현
        ↓
일반 차량(HV) 교통 흐름 구성
        ↓
자율주행 차량(AV)의 Observation / Action / Reward 정의
        ↓
BC 또는 RL 모델 학습
        ↓
학습 모델로 주행 데이터 수집
        ↓
충돌률·완주율·주행 영상·데이터셋 결과 확인
```

가장 중요한 것은 새로운 알고리즘을 많이 추가하는 것이 아니라, **팀이 만든 도로에서 일반 차량과 자율주행 차량이 안정적으로 주행하도록 전체 시스템을 완성하는 것**임.

---

## 2. 전체 코드 구조

```text
Artificial-Intelligence-and-Control-for-Autonomous-Driving/
│
├── env/
│   ├── road_config.py          # 도로·AV·일반 차량 설정
│   ├── road_builder.py         # 설정을 SUMO XML 파일로 변환
│   ├── mdp_config.py           # Observation·Action·Reward·Episode 설정
│   ├── sumo_env.py             # Gymnasium 환경과 TraCI 제어
│   └── sumo/                   # 자동 생성되는 SUMO 실행 파일
│
├── algorithms/
│   ├── ppo.py                  # PPO 강화학습 알고리즘
│   └── bc.py                   # Behavior Cloning 정책
│
├── utils/
│   ├── networks.py             # PPO 정책·가치 신경망
│   ├── buffer.py               # 학습 경험 저장
│   ├── evaluator.py            # 주행 성능 평가
│   └── logger.py               # CSV·TensorBoard 로그 기록
│
├── view_road.py                # 도로와 교통 흐름 확인
├── train.py                    # PPO 학습
├── train_bc.py                 # BC 학습
├── collect_pomdp_data.py       # 주행 데이터셋 수집
├── test.py                     # 학습 모델 주행 평가
│
├── data/                       # 수집한 데이터셋
└── results/                    # 학습 모델과 실험 결과
```

프로젝트 진행 시 모든 파일을 수정할 필요는 없음. 먼저 설정 파일로 구현 가능한 범위를 확인하고, 도로 구조나 MDP 자체를 바꿀 때 필요한 코드만 확장하는 것이 좋음.

| 변경할 내용 | 먼저 확인할 파일 | 추가 수정이 필요한 경우 |
|---|---|---|
| 도로 길이·차선 수·제한속도 | `env/road_config.py` | 복잡한 형상은 `env/road_builder.py` |
| 교차로·합류·분기·회전교차로 | `env/road_builder.py` | 연결 관계에 따라 `env/sumo_env.py` |
| 일반 차량 종류·비율·속도·교통량 | `env/road_config.py` | 새로운 Route/Flow는 `env/road_builder.py` |
| AV 시작 위치·차선·속도 | `env/road_config.py` | 여러 출발점은 `env/sumo_env.py` |
| 관측 범위와 관측 항목 | `env/mdp_config.py` | 항목 추가는 `env/sumo_env.py`의 `_get_obs()` |
| 가감속·차선변경 규칙 | `env/mdp_config.py` | 행동 종류 추가는 `env/sumo_env.py` |
| Reward와 종료 조건 | `env/mdp_config.py`, `train.py` | 계산 방식 변경은 `env/sumo_env.py`의 `step()` |
| PPO 설정·네트워크 크기 | `train.py` | 구조 변경은 `utils/networks.py` |
| BC 학습 설정 | `train_bc.py` 실행 옵션 | 구조 변경은 `algorithms/bc.py` |
| 평가 항목 | `utils/evaluator.py` | 출력 형식은 `test.py` |

---

## 3. 현재 제공된 기본 환경

현재 코드는 프로젝트를 시작하기 위한 **직선 고속도로 추월 예제**임.

```text
출발점 n0 ───────────── 단일 Edge e0 ───────────── 도착점 n1
                         3개 차선
```

기본 환경의 주요 특징은 다음과 같음.

- 길이 12 km의 직선 편도 도로
- 3개 차선과 33.33 m/s의 제한속도
- 빨간색 ego 차량 1대
- Krauss, IDM, EIDM, ACC 일반 차량 혼합
- ego보다 느린 일반 차량을 추월하는 학습 시나리오
- 주변 5개 차선 범위의 31차원 부분관측
- 가감속과 차선변경을 포함한 2차원 행동
- PPO 강화학습과 BC 모방학습 지원

이 환경은 코드 사용법을 보여주는 기준 예제이며, 실제 프로젝트 도로를 자동으로 만들어 주는 범용 도로 생성기는 아님.

> **중요:** 현재 `road_builder.py`는 노드 2개와 Edge 1개인 직선 도로를 생성함.  
> 삼거리, 사거리, 합류부, 로터리 등은 노드·Edge·Connection·Route 생성 부분을 팀 도로에 맞게 확장해야 함.

---

## 4. 코드가 실행되는 순서

### 4.1 도로 생성

```text
env/road_config.py
        ↓
env/road_builder.py의 build()
        ↓
highway.nod.xml + highway.edg.xml
        ↓
netconvert
        ↓
highway.net.xml
        ↓
highway.rou.xml + highway.sumocfg
```

`train.py`, `test.py`, `view_road.py`, `collect_pomdp_data.py`는 실행할 때마다 `road_builder.build()`를 호출함.

따라서 `env/sumo/` 안의 자동 생성 파일을 직접 수정해도 다음 실행에서 덮어쓸 수 있음. 영구적인 수정은 `road_config.py` 또는 `road_builder.py`에 작성하는 것을 권장함.

### 4.2 한 Episode의 실행

```text
env.reset()
    ↓
SUMO 실행 및 일반 차량 배치
    ↓
ego 차량 생성
    ↓
Observation 반환
    ↓
Policy가 Action 선택
    ↓
env.step(action)
    ↓
Action 적용 → SUMO 1스텝 진행 → 다음 Observation과 Reward 계산
    ↓
충돌 / 도착 / 시간초과 확인
```

Gymnasium 형식의 한 스텝은 다음 값을 반환함.

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated`: 충돌이나 목적지 도착처럼 주행 결과로 Episode가 끝남
- `truncated`: 최대 스텝 수에 도달하여 시간초과로 끝남
- `info`: 실제 차선변경, 속도, 차간거리 등 평가에 필요한 부가 정보

---

## 5. 도로와 교통 환경 수정

### 5.1 `env/road_config.py`

`road_config.py`는 도로와 차량의 물리적인 조건을 정의함.

```python
ROAD = {
    "length": 12000,
    "num_lanes": 3,
    "speed_limit": 33.33,
}
```

```python
EGO = {
    "accel": 5.4,
    "decel": 5.4,
    "max_speed": 33.33,
    "depart_lane": "center",
    "depart_speed": 10,
}
```

```python
TRAFFIC = {
    "max_speed": 16.0,
    "vehs_per_hour": 800,
    "depart_lane": "random",
    "depart_speed": 8.0,
    "controllers": {...},
    "prefill": {...},
}
```

프로젝트 도로를 구현할 때 확인할 내용은 다음과 같음.

- 실제 도로의 차선 수와 제한속도
- 각 방향의 진입·진출 지점
- 일반 차량의 Route와 방향별 교통량
- 차량추종모델의 종류와 구성 비율
- AV의 출발 지점과 목적지
- 학습 시작 직후에도 차량 상호작용이 발생하는지

`TRAFFIC["controllers"]`에서는 Krauss, IDM, EIDM, ACC 차량의 비율과 주행 특성을 변경할 수 있음. `probability`의 합은 코드에서 자동 정규화됨.

### 5.2 `env/road_builder.py`

현재 `build()`는 다음 SUMO 파일을 생성함.

| 파일 | 내용 |
|---|---|
| `highway.nod.xml` | 도로 끝점과 교차점인 Node |
| `highway.edg.xml` | Node를 연결하는 Edge와 차선 수·속도 |
| `highway.net.xml` | `netconvert`가 만든 최종 도로망 |
| `highway.rou.xml` | 차량 타입, Route, Traffic Flow |
| `highway.sumocfg` | SUMO가 읽는 전체 실행 설정 |
| `highway.gui.xml` | GUI 화면 확대와 재생 설정 |
| `highway.scenery.xml` | 도로 주변 시각 요소 |

복잡한 도로에서는 다음 요소를 추가로 설계해야 함.

```text
Node       : 교차점과 도로의 시작·끝 좌표
Edge       : 차량이 이동하는 도로 구간과 방향
Lane       : Edge별 차선 수와 속도
Connection : Edge 사이에서 어떤 차선으로 이동할 수 있는지
Route      : 출발 Edge부터 도착 Edge까지 이동 경로
Flow       : 각 Route에 차량을 얼마나 투입할지
```

예를 들어 사거리는 하나의 Edge만으로 만들 수 없음. 방향별 진입·진출 Edge, 교차로 Node, 좌회전·직진·우회전 Connection과 Route가 모두 필요함.

### 5.3 자동 생성 방식과 직접 제작 방식

팀의 도로 구현 방법은 크게 두 가지임.

**방법 A. `road_builder.py` 확장**

- Python 설정값으로 Node·Edge·Route 파일 생성
- 반복 실험과 팀 간 설정 공유가 쉬움
- 실행할 때마다 같은 도로를 재생성할 수 있음

**방법 B. SUMO `netedit` 또는 외부 도로 파일 사용**

- 복잡한 교차로와 Connection을 화면에서 편집하기 쉬움
- 이 경우 `road_builder.py`가 완성 파일을 덮어쓰지 않도록 실행 구조를 함께 변경해야 함
- 완성한 `.net.xml`, `.rou.xml`, `.sumocfg`를 저장소에 포함해야 함

어느 방식을 사용하더라도 팀원이 저장소를 새로 내려받아 같은 환경을 재현할 수 있어야 함.

---

## 6. AV의 POMDP 수정

### 6.1 `env/mdp_config.py`

`mdp_config.py`는 AV가 푸는 문제를 정의함.

| 설정 | 의미 |
|---|---|
| `SIMULATION` | 스텝 시간, Episode 길이, GUI 설정 |
| `OBSERVATION` | 감지 거리와 차량 밀도 계산 기준 |
| `ACTION` | 행동 차원, 차선변경 양자화·안전 조건 |
| `REWARD` | 속도, 충돌, 도착, 차간거리 등에 대한 보상 |

현재 Observation은 ego의 속도와 주변 차선의 선행·후행 차량, 차선 연결성, 밀도를 합친 31차원 벡터임.

```text
ego 속도 1개
    +
5개 차선 × (선행 거리·상대속도·후행 거리·상대속도) = 20개
    +
5개 차선 연결성
    +
5개 차선 밀도
    =
총 31차원
```

현재 Action은 다음 2개 값으로 구성됨.

```python
action = [accel_raw, lane_change_raw]
```

- `accel_raw`: `[-1, 1]`, 감속부터 가속까지의 제어 입력
- `lane_change_raw`: `[-1, 1]`, 오른쪽·유지·왼쪽 명령으로 변환

실제 도로에 신호등, 정지선, 경로 선택이 있다면 현재 관측과 행동만으로 충분한지 먼저 검토해야 함.

예:

- 신호등 상태와 정지선까지의 거리
- 교차로에서의 진행 방향
- 다음 Edge 또는 목표 Route 정보
- 횡단보도나 우선권 정보
- 차선별 진행 가능 방향

설정값만 추가하면 정책 입력에 자동 반영되는 것은 아님. 새로운 관측값은 `sumo_env.py`의 `_get_obs()`에서 계산하고, `observation_space`의 크기와 범위도 함께 맞춰야 함.

### 6.2 `env/sumo_env.py`

`SumoHighwayEnv`는 Python 학습 코드와 SUMO를 연결함.

| 함수 | 역할 |
|---|---|
| `__init__()` | 환경 설정, Observation·Action space 생성 |
| `reset()` | SUMO 시작, 차량 배치, ego 생성 |
| `_get_obs()` | 정책에 전달할 부분관측 계산 |
| `_quantize_lane_change()` | raw 차선 행동을 세 명령으로 변환 |
| `_lane_change_is_safe()` | 목표 차선의 앞뒤 안전거리 확인 |
| `_apply_lane_change()` | SUMO에 차선변경 명령 전달 |
| `get_privileged_state()` | 데이터셋용 SUMO 내부 상태 반환 |
| `step()` | 행동 적용, 시뮬레이션 진행, 보상·종료 계산 |
| `close()` | SUMO 연결 종료 |

교차로나 여러 Edge를 사용하는 경우 특히 다음 하드코딩 여부를 확인해야 함.

- ego가 생성되는 Route와 Edge
- 도착 판정 위치
- 도로 전체 길이를 이용한 완주 판정
- 현재 차선과 인접 차선 탐색 방식
- 차선이 합류·분기되는 지점의 연결성
- 충돌 이외의 실패 조건

도로는 화면에서 정상적으로 보여도, `reset()`과 `step()`이 새 Route 구조를 이해하지 못하면 학습 환경으로 사용할 수 없음.

---

## 7. Reward와 종료 조건 설계

기본 Reward에는 다음 항목이 포함됨.

```text
속도 보상
충돌 감점
목적지 도착 보너스
짧은 차간거리 감점
느린 앞차에 막힌 상태의 감점
차선변경 감점
실행할 수 없는 행동의 감점
```

`env/mdp_config.py`의 값은 기본값이며, PPO 실험에서는 `train.py`의 `REWARD_OVERRIDES`가 같은 이름의 값을 덮어씀.

```python
REWARD = {**REWARD_DEFAULTS, **REWARD_OVERRIDES}
```

따라서 Reward 실험 시 두 파일을 함께 확인해야 함. `test.py`와 데이터 수집 코드도 `train.py`의 `REWARD`를 사용함.

실제 도로에 따라 다음 Reward가 추가될 수 있음.

- 신호 위반 감점
- 잘못된 차선 진입 감점
- 경로 이탈 감점
- 정지선 준수 보상
- 불필요한 정차나 급가감속 감점
- 목표 Route 또는 출구 도착 보상

Reward를 추가할 때는 값만 정의하지 말고 `sumo_env.py`의 `step()`에서 실제로 계산되는지 확인해야 함.

> **주의:** Reward가 높아졌다는 사실만으로 주행이 좋아졌다고 판단하지 않음.  
> 충돌률, 완주율, 시간초과율과 SUMO GUI의 실제 행동을 함께 확인함.

---

## 8. 프로젝트 실행 순서

### Step 1. 도로만 확인

```bash
python view_road.py
```

도로망을 `netedit`으로 확인하려면 다음을 실행함.

```bash
python view_road.py --netedit
```

화면 없이 SUMO 설정 오류와 차량 투입 여부만 확인할 수도 있음.

```bash
python view_road.py --nogui --seconds 120
```

이 단계에서는 모델을 학습하지 않음. 먼저 다음을 확인함.

- 모든 Edge와 Connection이 의도한 방향으로 연결되는가?
- 모든 Route가 실제로 주행 가능한가?
- 차량이 특정 Junction에서 멈추거나 사라지지 않는가?
- 일반 차량의 교통량과 속도가 자연스러운가?
- 차량이 생성되지 못하고 대기하는 구간은 없는가?

### Step 2. 환경의 한 Episode 확인

기본 정책으로 짧은 데이터를 수집하면 `reset()`과 `step()`의 전체 흐름을 점검할 수 있음.

```bash
python collect_pomdp_data.py --policy keep-lane --episodes 1 --gui --name env_check
```

다음 항목을 확인함.

- ego가 올바른 위치와 Route에서 생성되는가?
- Observation의 차원과 값 범위가 일정한가?
- 가감속과 차선변경 명령이 실제로 적용되는가?
- 충돌, 도착, 시간초과가 올바르게 구분되는가?
- Reward가 의도한 상황에서 증가하거나 감소하는가?

### Step 3. 작은 규모로 학습 확인

최종 학습 전에 `TOTAL_TIMESTEPS`, Episode 수, epoch 등을 줄여 코드 전체가 끝까지 실행되는지 확인함.

PPO:

```bash
python train.py
```

BC:

```bash
python train_bc.py --data data/dataset.npz --epochs 5 --out-dir results/bc_check
```

작은 실험에서 모델, CSV, TensorBoard 파일이 정상적으로 저장된 뒤 본 학습을 진행함.

### Step 4. 본 학습과 실험 기록

한 번에 여러 값을 바꾸지 않고, 실험마다 변경 목적을 명확히 기록함.

```text
Experiment A: 기본 환경
Experiment B: 교통량 변경
Experiment C: Reward 변경
Experiment D: 학습률 변경
```

PPO 결과 폴더에는 학습 당시의 `train.py`, `road_config.py`, `mdp_config.py`가 복사됨. 모델 성능을 비교할 때 현재 파일이 아니라 각 결과 폴더의 설정 스냅샷을 확인함.

### Step 5. 모델 평가

GUI로 주행 확인:

```bash
python test.py results/run_모델폴더/model.pt --episodes 5
```

화면 없이 여러 Episode 평가:

```bash
python test.py results/run_모델폴더/model.pt --episodes 100 --nogui
```

평가에서는 다음 지표를 확인함.

- 충돌률
- 완주율
- 시간초과율
- 평균 속도
- 평균·최소 차간거리
- Episode당 차선변경 횟수
- 평균 누적 보상
- 평균 Episode 길이

프로젝트 기본 목표인 **Collision Rate ≤ 5%**를 확인하려면 충분한 수의 Episode로 평가해야 함.

### Step 6. 최종 주행 데이터 수집

학습된 PPO 모델을 사용한 예:

```bash
python collect_pomdp_data.py \
  --model results/run_모델폴더/model.pt \
  --episodes 100 \
  --workers 4 \
  --name team_final
```

출력 파일:

```text
data/team_final.npz
data/team_final.jsonl.gz
```

- NPZ: NumPy와 PyTorch에서 바로 사용하기 쉬운 고정 크기 배열
- JSONL.GZ: 부분관측뿐 아니라 privileged state와 감지 차량 정보까지 포함

GUI 사용 시에는 워커 수가 1로 고정됨.

```bash
python collect_pomdp_data.py \
  --model results/run_모델폴더/model.pt \
  --episodes 3 \
  --gui \
  --name team_final_gui
```

데이터 수집 전에 모델 학습 당시의 도로·관측·행동 구성이 현재 환경과 같은지 확인함.

---

## 9. 학습 코드의 역할

### 9.1 PPO: `train.py`와 `algorithms/ppo.py`

`train.py`는 환경과 PPO 에이전트를 만들고 학습을 시작하는 진입점임.

```text
train.py
    ↓
SumoHighwayEnv 생성
    ↓
PPO + RolloutBuffer 생성
    ↓
주행 경험 수집
    ↓
GAE 계산
    ↓
PPO update
    ↓
주기적 평가와 로그 저장
```

주로 수정하는 값은 다음과 같음.

- `TOTAL_TIMESTEPS`
- `REWARD_OVERRIDES`
- `HPARAMS`
- `EVAL_INTERVAL`, `EVAL_EPISODES`
- `RUN_NAME`

새로운 도로를 만들었다고 바로 `algorithms/ppo.py`를 수정할 필요는 없음. 먼저 환경과 Reward가 정상인지 검증한 뒤 알고리즘 변경을 진행함.

### 9.2 BC: `train_bc.py`와 `algorithms/bc.py`

BC는 저장된 `(observation, action_raw)`을 이용해 수집 정책의 행동을 모방함.

```text
학습된 PPO 또는 다른 정책
        ↓
collect_pomdp_data.py
        ↓
NPZ 데이터
        ↓
train_bc.py
        ↓
BCPolicy
        ↓
model.pt
```

현재 BC는 Reward를 직접 사용하지 않음. 수집한 정책의 좋은 행동뿐 아니라 잘못된 행동도 함께 학습할 수 있으므로, 데이터 품질과 행동 분포를 먼저 확인해야 함.

---

## 10. 팀 작업 분담 예시

팀 인원과 도로 난이도에 따라 역할을 조정할 수 있음.

| 역할 | 주요 작업 | 관련 파일 |
|---|---|---|
| 도로 구축 | 실제 도로 조사, Node·Edge·Connection·Route 구현 | `road_config.py`, `road_builder.py` |
| 교통 환경 | 일반 차량 종류·속도·교통량·경로 구성 | `road_config.py`, `road_builder.py` |
| AV 환경 | Observation·Action·Reward·종료 조건 구현 | `mdp_config.py`, `sumo_env.py` |
| 모델 학습 | PPO/BC 학습, 하이퍼파라미터 실험 | `train.py`, `train_bc.py`, `algorithms/` |
| 평가·데이터 | 반복 평가, 그래프, 데이터셋, 영상 정리 | `test.py`, `collect_pomdp_data.py`, `utils/` |

파일만 나누어 작업하지 말고 다음 인터페이스를 팀 전체가 함께 확인해야 함.

```text
도로 Route ↔ ego 생성 위치
도로 차선 ↔ Observation
Action ↔ 실제 SUMO 제어
종료 조건 ↔ 목적지
Reward ↔ 평가 지표
학습 환경 ↔ 평가·데이터 수집 환경
```

---

## 11. 프로젝트 진행 시 자주 발생하는 문제

### Q1. 도로 파일을 수정했는데 다시 원래대로 돌아오는 경우

`train.py`와 `test.py`가 실행될 때 `road_builder.build()`가 `env/sumo/` 파일을 다시 생성함.

직접 수정한 XML을 유지하려면 다음 중 하나를 선택함.

- 수정 내용을 `road_builder.py`에 반영
- 완성된 SUMO 파일을 사용하도록 `build()`와 실행 코드를 변경

### Q2. GUI에서는 도로가 정상인데 ego가 생성되지 않는 경우

다음을 확인함.

- ego Route에 존재하지 않는 Edge가 포함되어 있지 않은가?
- 시작 차선 번호가 해당 Edge의 차선 범위 안에 있는가?
- 시작 위치에 다른 차량이 있어 삽입이 거부되지 않는가?
- `reset()`에서 사용하는 Route ID가 새 도로의 Route와 같은가?

### Q3. 차량이 교차로에서 멈추는 경우

- Edge 사이 Connection이 존재하는가?
- Route가 연결된 Edge 순서로 작성되었는가?
- 신호 프로그램 또는 우선권 설정이 올바른가?
- 차량 수가 교차로 용량보다 지나치게 많지 않은가?

### Q4. 도로를 바꾼 뒤 Observation 오류가 발생하는 경우

- `observation_space`와 `_get_obs()`의 실제 길이가 같은가?
- 주변 차량이 없을 때도 고정 길이 값을 반환하는가?
- 존재하지 않는 차선과 끊기는 차선을 구분하는가?
- 모든 값이 선언한 범위 안에 있는가?

### Q5. 모델은 실행되지만 성능이 이상한 경우

- 학습 당시와 현재 Observation의 순서와 의미가 같은가?
- 학습 당시와 현재 네트워크 구조가 같은가?
- 결과 폴더의 설정 스냅샷과 현재 설정을 비교했는가?
- Reward가 의도하지 않은 행동에 더 큰 값을 주지 않는가?
- 일반 차량의 교통량이 너무 적거나 많지 않은가?

### Q6. Reward는 증가하지만 충돌률이 줄지 않는 경우

Reward 항 사이의 상대 크기를 확인함. 긴 시간 동안 얻는 속도 보상이 한 번의 충돌 감점보다 지나치게 크면 충돌을 감수하는 정책이 만들어질 수 있음.

Reward 곡선만 보지 말고 충돌률·완주율과 실제 주행을 함께 평가함.

---

## 12. 권장 프로젝트 진행 순서

아래 순서를 지키면 문제의 원인을 구분하기 쉬움.

```text
1. 실제 도로 조사와 단순화 범위 결정
2. Node / Edge / Lane / Connection 구현
3. 모든 Route 단독 주행 확인
4. 일반 차량 교통 흐름 확인
5. ego 생성과 목적지 도착 확인
6. Observation 값 확인
7. Action 적용 확인
8. Reward와 종료 조건 확인
9. 작은 규모 학습
10. 본 학습과 하이퍼파라미터 비교
11. 100 Episode 이상 최종 평가
12. 데이터셋·영상·보고서 정리
```

도로와 환경이 완성되기 전에 긴 학습을 시작하지 않음. 환경 오류는 학습 시간을 늘려도 해결되지 않음.

---

## 13. 팀 프로젝트 체크리스트

### 도로와 교통

- [ ] 실제 한국 도로에서 참고한 특징이 명확한가?
- [ ] Node, Edge, Lane, Connection이 올바르게 연결되는가?
- [ ] 모든 진입·진출 Route를 차량이 완주할 수 있는가?
- [ ] 일반 차량이 자연스럽게 생성되고 이동하는가?
- [ ] 특정 위치에서 무한 정체나 비정상 정지가 발생하지 않는가?
- [ ] 차량 간 의미 있는 상호작용이 발생하는가?

### AV 환경

- [ ] ego가 매 Episode 정상적으로 생성되는가?
- [ ] Observation의 크기와 값 범위가 항상 일정한가?
- [ ] Action이 실제 SUMO 차량에 적용되는가?
- [ ] Reward의 각 항목이 의도한 상황에서 계산되는가?
- [ ] 충돌·도착·시간초과 종료 조건이 정확한가?

### 학습과 평가

- [ ] 작은 규모 학습으로 전체 파이프라인을 먼저 확인했는가?
- [ ] 실험마다 변경한 설정과 목적을 기록했는가?
- [ ] 모델과 설정 스냅샷이 함께 저장되었는가?
- [ ] Reward뿐 아니라 충돌률과 완주율을 비교했는가?
- [ ] GUI에서 실제 주행 행동을 확인했는가?
- [ ] 충분한 Episode로 최종 성능을 평가했는가?

### 결과물

- [ ] 최종 도로 코드와 SUMO 파일이 저장소에 포함되었는가?
- [ ] 다른 환경에서 저장소를 내려받아 실행할 수 있는가?
- [ ] 학습 모델과 실행 방법이 정리되어 있는가?
- [ ] 주행 데이터셋이 정상적으로 저장되는가?
- [ ] 결과 그래프와 주행 영상이 준비되어 있는가?
- [ ] 프로젝트 페이지와 보고서에서 팀의 변경 내용을 설명하는가?

---

## 14. 가장 기억해야 할 것

프로젝트 코드는 다음 세 층으로 나누어 생각하면 이해하기 쉬움.

```text
Road & Traffic
road_config.py + road_builder.py
        ↓
Driving Environment
mdp_config.py + sumo_env.py
        ↓
Learning & Evaluation
train.py + algorithms/ + test.py + collect_pomdp_data.py
```

위쪽의 도로 구조가 바뀌면 아래쪽 환경의 ego 생성, 관측, Reward, 종료 조건도 함께 확인해야 함. 모델 학습은 이 연결이 모두 정상적으로 동작한 다음 단계임.

> **최종 목표:**  
> **팀이 직접 구현한 대한민국 도로 위에서 일반 차량이 자연스럽게 이동하고, 학습한 자율주행 차량이 낮은 충돌률로 목적지까지 주행하도록 만들기.**
