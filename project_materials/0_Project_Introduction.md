# 🚗 Course Project: 대한민국 도로 구조 기반 자율주행 강화학습

## 1. 프로젝트 주제

> **대한민국의 실제 도로 구조를 SUMO 환경에 구현하고, 해당 도로에서 주행하는 자율주행 차량의 의사결정 모델을 구축한다.**

이번 프로젝트의 핵심은 단순히 강화학습 알고리즘의 성능을 높이는 것이 아닙니다.

학생들이 직접 **한국의 실제 도로 구조를 관찰하고**, 이를 시뮬레이션 환경으로 구현한 뒤,  
일반 차량과 자율주행 차량이 실제로 주행할 수 있는 환경을 만드는 것이 가장 중요한 목표입니다.

최종적으로는 다음 과정을 하나의 프로젝트 안에서 수행합니다.

```text
실제 도로 선정
    ↓
SUMO 도로 구조 구축
    ↓
일반 차량(HV) 주행 환경 구축
    ↓
자율주행 차량(AV) 상태·행동·보상 모델링
    ↓
Online RL 학습
    ↓
주행 데이터셋 수집
    ↓
BC / Offline RL 학습 및 성능 평가
```

---

# 2. 최종 목표

프로젝트가 끝났을 때 각 팀은 다음과 같은 결과물을 완성해야 합니다.

> **"우리나라의 실제 도로를 직접 모델링하고, 그 위에서 일반 차량과 자율주행 차량이 정상적으로 주행하도록 만든다."**

즉,

- 한국의 실제 도로 구조를 반영한 SUMO 환경을 직접 구축하고
- 일반 차량이 자연스럽게 주행하도록 설정하고
- 자율주행 차량의 관측(State/Observation), 행동(Action), 보상(Reward)을 정의하고
- 강화학습을 통해 자율주행 차량을 학습시키고
- 학습 과정에서 생성된 데이터를 저장하여
- 최종적으로 BC 또는 Offline RL 학습까지 진행합니다.

---

# 3. 어떤 도로를 만들면 되나요?

## 기본 원칙

각 팀은 **대한민국에 실제로 존재하는 도로 또는 교차로 구조**를 하나 선정합니다.

단순한 직선 도로보다는 차량 간 상호작용이 발생할 수 있는 구조를 권장합니다.

예를 들면 다음과 같습니다.

- 삼거리
- 사거리
- 오거리
- 로터리 / 회전교차로
- 합류 구간
- 분기 구간
- 차로 감소 구간
- 복합 교차로
- 비정형 교차로

도로가 반드시 실제 도로와 **1:1로 완벽하게 동일할 필요는 없습니다.**

다만 실제 도로의 특징을 참고하여

- 차로 수
- 교차로 형태
- 진입 / 진출 방향
- 회전 구조
- 차선 연결

등을 최대한 자연스럽게 구현하는 것을 목표로 합니다.

---

# 4. 예시 도로

아래와 같은 도로들을 예시로 생각할 수 있습니다.

## 예시 1. 팔달문 일대

**특징**

- 로터리 형태
- 여러 방향의 차량 진입
- 삼거리 구조 포함
- 차량 간 상호작용이 많음

이와 같은 환경에서는 자율주행 차량이

- 로터리 진입
- 주변 차량 확인
- 적절한 속도 조절
- 원하는 출구 선택

등의 의사결정을 수행할 수 있습니다.

---

## 예시 2. 일반적인 삼거리

**특징**

- 상대적으로 구현 난이도가 낮음
- 직진 / 좌회전 / 우회전 차량 간 상호작용 발생
- 자율주행 차량의 기본적인 의사결정 문제를 구성하기 좋음

처음 SUMO 환경을 구축하는 팀에게 적합합니다.

---

## 예시 3. 숭례문 인근 오거리

**특징**

- 여러 방향에서 차량이 진입
- 복잡한 차선 연결
- 다양한 차량 경로 구성 가능
- 복잡한 도로 구조를 설계하기에 적합

도로 구조 자체의 완성도를 높이고 싶은 팀에게 적합합니다.

---

## 💡 다른 지역을 선택해도 됩니다!

위 도로들은 단순한 예시입니다.

특히 외국인 학생이라면 자신이 흥미롭게 생각하는 한국의 지역이나 관광지를 선택해도 좋습니다.

예:

- 광화문
- 강남역
- 홍대입구
- 부산 해운대
- 제주도 주요 교차로
- 동대문
- 경복궁 인근
- 서울역
- 남산 주변

> **중요:** 유명한 장소를 선택하는 것보다,  
> **흥미로운 차량 상호작용을 만들 수 있는 도로 구조를 선택하는 것이 더 중요합니다.**

---

# 5. 프로젝트 진행 과정

프로젝트는 크게 **5단계**로 진행합니다.

---

## Step 1. 도로 구조 구축

가장 먼저 SUMO에서 도로 환경을 구축합니다.

학생들이 직접 구현해야 하는 요소의 예시는 다음과 같습니다.

- Edge
- Lane
- Junction
- Connection
- Route
- Traffic flow

필요하다면 다음과 같은 요소를 추가할 수도 있습니다.

- 교통신호
- 우선순위 도로
- 회전교차로
- 다중 차선
- 차로 감소
- 합류 / 분기 구간

### Step 1의 목표

```text
내가 만든 도로에서 SUMO 차량이 정상적으로 주행한다.
```

이 단계에서는 아직 강화학습이 없어도 됩니다.

먼저 **시뮬레이션 자체가 안정적으로 동작하는지 확인하는 것**이 중요합니다.

---

## Step 2. 일반 차량(Human-driven Vehicle) 구축

도로가 완성되면 일반 차량들을 추가합니다.

일반 차량은 기본적으로 **IDM 기반 controller** 등을 사용할 수 있습니다.

예를 들어 일반 차량들은 다음과 같은 행동을 수행할 수 있습니다.

- 앞 차량과 안전거리 유지
- 가속 / 감속
- 정체 발생 시 감속
- 필요할 경우 차선 변경

### 확인해야 할 사항

- 차량이 정상적으로 생성되는가?
- 차량끼리 지나치게 충돌하지 않는가?
- 차량 흐름이 자연스러운가?
- 특정 위치에서 차량이 무한정 정체되지 않는가?
- 모든 Route가 정상적으로 연결되어 있는가?

---

## Step 3. 자율주행 차량의 POMDP / Reward 모델링

이제 일반 차량 중 일부를 **자율주행 차량(AV)** 으로 변경합니다.

자율주행 차량이 강화학습을 수행하려면 다음 요소를 정의해야 합니다.

### 3-1. Observation / State

자율주행 차량이 주변 환경에서 어떤 정보를 볼 수 있는지 정의합니다.

예:

```text
Ego vehicle
- 현재 속도
- 현재 가속도
- 현재 차선
- 목적지까지의 거리

Surrounding vehicles
- 앞 차량과의 거리
- 앞 차량의 상대 속도
- 뒤 차량과의 거리
- 인접 차선 차량과의 거리
```

모든 차량 정보를 사용해야 하는 것은 아닙니다.

실제 자율주행 차량처럼 **제한된 범위의 주변 차량만 관측하도록 구성**해도 좋습니다.

---

### 3-2. Action

자율주행 차량이 선택할 수 있는 행동을 정의합니다.

예:

```text
Longitudinal action
- Acceleration
- Deceleration

Lateral action
- Lane change left
- Keep lane
- Lane change right
```

프로젝트 난이도에 따라 행동 공간을 단순화해도 됩니다.

예를 들어 처음에는

```text
Action = Acceleration
```

만 학습시키고, 차선 변경은 SUMO의 기본 controller를 사용할 수도 있습니다.

---

### 3-3. Reward

자율주행 차량이 어떤 행동을 좋은 행동으로 판단할지 정의합니다.

예:

```text
+ 목적지 도착
+ 높은 평균 속도
+ 원활한 도로 통과

- 차량 충돌
- 지나치게 낮은 속도
- 급가속 / 급감속
- 도로 이탈
```

예를 들어 간단한 Reward는 다음과 같이 구성할 수 있습니다.

```text
Reward
= Speed Reward
+ Goal Reward
- Collision Penalty
```

> Reward를 복잡하게 만드는 것이 목표는 아닙니다.  
> **자신이 만든 도로에서 차량이 어떤 행동을 해야 하는지 생각하고 이를 Reward로 표현하는 것**이 중요합니다.

---

## Step 4. Online Reinforcement Learning

구축한 환경에서 자율주행 차량을 Online RL로 학습합니다.

기본적으로 수업에서 제공하는 강화학습 알고리즘을 활용하면 됩니다.

학생들이 직접 새로운 RL 알고리즘을 구현하는 것은 필수가 아닙니다.

### 기본 목표

```text
환경 실행
→ AV가 행동 선택
→ SUMO simulation 진행
→ Reward 계산
→ Transition 저장
→ RL model update
```

이 과정을 반복합니다.

### 추가 도전

프로젝트 진행 능력이 충분한 경우 다음과 같은 추가 실험도 가능합니다.

- 다른 RL 알고리즘 적용
- Reward 구조 변경
- Observation 변경
- Action space 변경
- 여러 대의 AV 사용
- Curriculum learning
- 안전성 제약 추가

이러한 내용은 **Bonus 요소**로 평가할 수 있습니다.

---

# 6. 주행 데이터셋 구축

Online RL 학습 과정에서 다음과 같은 데이터를 저장합니다.

```text
(state, action, reward, next_state, done)
```

또는 POMDP 환경에서는

```text
(observation, action, reward, next_observation, done)
```

형태로 저장할 수 있습니다.

예:

| Episode | Step | Observation | Action | Reward | Next Observation | Done |
|---|---:|---|---|---:|---|---|
| 1 | 0 | ... | accelerate | 0.42 | ... | False |
| 1 | 1 | ... | keep | 0.51 | ... | False |
| 1 | 2 | ... | brake | -0.13 | ... | False |

### 데이터셋 구축 시 확인할 것

- 데이터가 실제로 저장되고 있는가?
- State / Action / Reward가 올바르게 기록되는가?
- Episode가 구분되어 있는가?
- 학습에 사용할 수 있는 형태로 저장되어 있는가?

---

# 7. 데이터 기반 Policy 학습

수집한 데이터를 이용하여 새로운 Policy를 학습합니다.

기본적으로 다음 두 가지 방향을 사용할 수 있습니다.

## Option A. Behavior Cloning (BC)

수집된 데이터에서

```text
Observation → Action
```

관계를 지도학습으로 학습합니다.

즉, 데이터에 기록된 행동을 그대로 따라하도록 학습하는 방식입니다.

---

## Option B. Offline Reinforcement Learning

수집한 데이터만 이용하여 Policy를 학습합니다.

Online RL과 달리 새로운 환경 interaction 없이

```text
이미 수집된 dataset
```

만을 이용하여 학습합니다.

---

## 중요

이번 프로젝트에서는 **누가 더 복잡한 Offline RL 알고리즘을 구현했는지**를 중심으로 평가하지 않습니다.

기본 제공 알고리즘을 사용해도 충분합니다.

더 중요한 것은

```text
좋은 도로 환경 구축
+
정상적인 차량 주행
+
올바른 데이터 수집
+
학습된 정책의 정상 작동
```

입니다.

---

# 8. 프로젝트에서 가장 중요한 부분 ⭐

이번 프로젝트에서 가장 중요한 부분은 **도로 구조와 환경 구축**입니다.

즉,

> "누가 가장 어려운 강화학습 알고리즘을 구현했는가?"

보다

> "누가 한국 도로의 특징을 잘 반영하면서 흥미롭고 안정적인 자율주행 환경을 만들었는가?"

가 더 중요합니다.

### 좋은 프로젝트의 예

- 실제 한국 도로의 특징이 잘 나타남
- 여러 방향의 차량 흐름이 존재함
- 차량 간 의미 있는 interaction이 발생함
- SUMO simulation이 안정적으로 실행됨
- 차량이 특정 위치에서 비정상적으로 멈추지 않음
- 학습용 데이터가 정상적으로 생성됨
- 자율주행 차량이 낮은 충돌률로 주행함

---

# 9. 평가 기준

평가는 크게 다음 요소를 기준으로 진행합니다.

| 평가 항목 | 평가 내용 |
|---|---|
| **도로 구조 구현** | 실제 한국 도로의 특징을 얼마나 잘 반영했는가 |
| **도로 구조의 복잡성 / 완성도** | 교차로, 차선, 합류, 분기 등 의미 있는 구조가 존재하는가 |
| **Simulation 안정성** | 차량이 정상적으로 주행하며 비정상적인 정체나 오류가 발생하지 않는가 |
| **일반 차량 구성** | IDM 등 일반 차량 controller가 정상적으로 작동하는가 |
| **AV 모델링** | Observation / Action / Reward가 적절하게 정의되었는가 |
| **Online RL** | 자율주행 차량 학습이 정상적으로 수행되는가 |
| **Collision Rate** | 최종 자율주행 차량의 충돌률이 충분히 낮은가 |
| **Dataset 구축** | State / Action / Reward 등의 데이터가 정상적으로 저장되었는가 |
| **BC / Offline RL** | 수집한 데이터를 이용한 Policy 학습 및 평가가 수행되었는가 |
| **추가 도전** | 추가 알고리즘, 환경 개선, 분석 등을 수행했는가 |

---

## 충돌률 기준

기본 목표는

> **Collision Rate ≤ 5%**

입니다.

충돌률은 예를 들어 다음과 같이 계산할 수 있습니다.

```text
Collision Rate
= 충돌이 발생한 Evaluation Episode 수
  / 전체 Evaluation Episode 수
```

예:

```text
100 Episode 평가
5 Episode에서 Collision 발생

→ Collision Rate = 5%
```

---

# 10. 정성적 평가

도로 구조는 단순한 숫자만으로 평가하기 어렵기 때문에 **정성적 평가**도 함께 진행할 수 있습니다.

예를 들어 다른 팀의 도로를 보고 다음 내용을 평가할 수 있습니다.

- 실제 도로처럼 보이는가?
- 도로 구조가 흥미로운가?
- 차량 interaction이 충분히 발생하는가?
- 구현 난이도가 적절한가?
- 시뮬레이션이 안정적인가?
- 자율주행 문제로서 의미가 있는가?

---

# 11. Bonus

다음 요소는 필수가 아니며 추가 점수 요소로 활용될 수 있습니다.

### Algorithm Bonus

- 추가 RL 알고리즘 적용
- 여러 Offline RL 알고리즘 비교
- BC와 Offline RL 비교
- Reward 설계 비교

### Environment Bonus

- 복잡한 회전교차로
- 다중 교차로
- 비정형 오거리
- 합류 + 분기 복합 도로
- 실제 교통량을 참고한 traffic flow

### Analysis Bonus

- Reward curve 분석
- Collision 위치 분석
- 차량 속도 분포 분석
- Action 분포 분석
- State visitation 분석
- Online RL과 Offline RL 결과 비교

> Bonus를 하지 않아도 기본 프로젝트를 완성할 수 있습니다.

---

# 12. 권장 프로젝트 진행 순서

처음부터 RL을 학습시키려고 하지 마세요.

아래 순서대로 하나씩 확인하는 것을 권장합니다.

## Phase 1 — Road

```text
도로 생성
↓
SUMO-GUI에서 직접 실행
↓
모든 차선과 Junction 확인
```

### 완료 조건

> 차량이 없어도 도로 구조가 정상적으로 보인다.

---

## Phase 2 — Traffic

```text
Route 생성
↓
일반 차량 생성
↓
IDM controller 적용
↓
Traffic simulation 실행
```

### 완료 조건

> 일반 차량들이 충돌 없이 정상적으로 도로를 주행한다.

---

## Phase 3 — AV

```text
AV 추가
↓
Observation 정의
↓
Action 정의
↓
Reward 정의
```

### 완료 조건

> Python에서 AV의 State / Action / Reward 값을 확인할 수 있다.

---

## Phase 4 — Online RL

```text
Online interaction
↓
Replay Buffer
↓
RL Training
↓
Evaluation
```

### 완료 조건

> 학습 전보다 AV의 주행 성능이 개선된다.

---

## Phase 5 — Dataset

```text
Simulation
↓
Trajectory 저장
↓
Dataset 생성
```

### 완료 조건

> `(s, a, r, s', done)` 데이터가 정상적으로 저장된다.

---

## Phase 6 — Offline Learning

```text
Dataset
↓
BC 또는 Offline RL
↓
Policy
↓
Simulation Evaluation
```

### 완료 조건

> Dataset만을 이용해 학습한 Policy가 SUMO에서 실제로 차량을 제어한다.

---

# 13. 권장 역할 분담

팀 프로젝트의 경우 다음과 같이 역할을 나눌 수 있습니다.

### Road / SUMO

- 실제 도로 조사
- SUMO network 구현
- Junction / Lane / Route 구성

### Environment

- TraCI interaction
- Observation 구성
- Reward 계산
- Episode 관리

### RL

- Online RL training
- Replay buffer
- Policy evaluation

### Dataset / Offline Learning

- Dataset 저장
- Data preprocessing
- BC / Offline RL
- 결과 분석

단, 각자 자신의 파트만 알고 끝나는 것이 아니라  
**최종적으로 전체 pipeline이 어떻게 연결되는지는 모든 팀원이 이해해야 합니다.**

---

# 14. 최종 제출물 예시

각 팀은 최종적으로 다음과 같은 결과를 제출할 수 있습니다.

```text
project/
│
├── README.md
│
├── network/
│   ├── *.net.xml
│   ├── *.rou.xml
│   └── *.sumocfg
│
├── env/
│   └── environment.py
│
├── controller/
│   ├── hv_controller.py
│   └── av_controller.py
│
├── training/
│   ├── train_online.py
│   ├── train_bc.py
│   └── train_offline.py
│
├── dataset/
│   └── trajectory.*
│
├── evaluation/
│   └── evaluate.py
│
└── results/
    ├── figures/
    └── videos/
```

파일 구조는 팀별로 달라도 됩니다.

---

# 15. 최종 시연에서 보여주면 좋은 내용

최종 발표에서는 최소한 다음 내용을 보여주는 것을 권장합니다.

### ① 실제 도로

Google Maps / Kakao Map / Naver Map 등의 자료를 이용해  
어떤 실제 도로를 참고했는지 설명합니다.

### ② 구현한 SUMO 도로

실제 도로와 구현한 SUMO 환경을 비교합니다.

### ③ 일반 차량 주행

IDM 기반 차량들이 정상적으로 주행하는 모습을 보여줍니다.

### ④ 자율주행 차량

학습된 AV가 실제 환경에서 주행하는 모습을 보여줍니다.

### ⑤ 학습 결과

예:

- Episode Reward
- Collision Rate
- Success Rate

### ⑥ Dataset

수집된 데이터의 형태와 규모를 보여줍니다.

### ⑦ BC / Offline RL 결과

수집한 데이터로 학습한 Policy가 실제로 주행하는 모습을 확인합니다.

---

# 16. 프로젝트 성공 기준

아래 항목이 모두 만족되면 프로젝트의 기본 목표를 달성한 것입니다.

- [ ] 대한민국의 실제 도로를 참고하여 SUMO 도로를 직접 구축하였다.
- [ ] 일반 차량이 정상적으로 주행한다.
- [ ] 자율주행 차량의 Observation을 정의하였다.
- [ ] 자율주행 차량의 Action을 정의하였다.
- [ ] Reward function을 정의하였다.
- [ ] Online RL 학습을 수행하였다.
- [ ] Collision Rate가 5% 이하이다.
- [ ] 주행 데이터를 Dataset으로 저장하였다.
- [ ] Dataset을 이용하여 BC 또는 Offline RL을 수행하였다.
- [ ] 학습된 Policy를 SUMO 환경에서 다시 평가하였다.

---

# 17. 가장 기억해야 할 것

이번 프로젝트의 목표는 **최신 강화학습 알고리즘을 새로 개발하는 것**이 아닙니다.

가장 중요한 것은

> **직접 만든 대한민국 도로에서 차량들이 실제로 움직이게 만드는 것**

입니다.

도로를 직접 만들고,  
일반 차량을 움직이고,  
자율주행 차량을 학습시키고,  
그 과정에서 데이터를 수집하고,  
다시 그 데이터를 이용해 차량을 학습시키는 전체 과정을 경험하는 것이 프로젝트의 핵심입니다.

```text
Build the Road
      ↓
Run the Traffic
      ↓
Train the AV
      ↓
Collect the Data
      ↓
Learn from the Data
```

### 최종 목표

> **"내가 만든 한국 도로 위에서 내가 학습시킨 자율주행 차량이 스스로 주행하도록 만들기."**
