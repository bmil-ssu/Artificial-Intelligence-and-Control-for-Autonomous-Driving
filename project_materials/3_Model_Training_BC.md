# 3. Model Training — Behavior Cloning

> **목표:** `collect_pomdp_data.py`로 수집한 주행 데이터를 이용하여, SUMO를 다시 실행하지 않고 `train_bc.py`에서 Behavior Cloning(BC) Policy를 학습함

---

# 1. 이번 단계에서 무엇을 하나요?

앞 단계에서 SUMO 도로 환경과 Observation / Action 구조를 구성했음

이번 단계에서는 다음 두 과정을 수행함

```text
1. 학습용 주행 데이터 수집
2. 수집한 데이터로 BC Policy 학습
```

현재 코드 기준 전체 흐름은 다음과 같음

```text
기존 Policy로 SUMO 주행
        ↓
collect_pomdp_data.py
        ↓
NPZ / JSONL Dataset 저장
        ↓
train_bc.py
        ↓
algorithms/bc.py의 BCPolicy 학습
        ↓
model.pt 저장
        ↓
test.py에서 SUMO 주행 평가
```

BC 학습 자체에서는 SUMO를 실행하지 않음

이미 저장된 Observation과 Action을 불러와 일반적인 supervised learning 방식으로 학습함

---

# 2. 관련 코드

현재 BC 학습에 직접 관련된 파일은 다음과 같음

```text
Artificial-Intelligence-and-Control-for-Autonomous-Driving/
│
├── collect_pomdp_data.py
├── train_bc.py
├── test.py
└── algorithms/
    ├── __init__.py
    └── bc.py
```

| 파일 | 역할 | 주로 확인할 내용 |
|---|---|---|
| `collect_pomdp_data.py` | SUMO에서 주행 데이터를 수집하여 NPZ/JSONL로 저장 | 수집 episode 수, 저장 경로, 사용 Policy |
| `train_bc.py` | Dataset을 읽고 BC 모델을 학습 | epoch, batch size, learning rate, validation split |
| `algorithms/bc.py` | 실제 BC Policy Network 정의 | MLP 구조, acceleration head, lane-change head |
| `test.py` | 학습된 BC Policy를 SUMO에 넣어 평가 | model path, `--algorithm bc`, GUI 사용 여부 |

---

# 3. Step 1 — 주행 데이터 수집

현재 사용 중인 데이터 수집 명령은 다음과 같음

```bash
python collect_pomdp_data.py \
    --model results/run_20260907_202220/model.pt \
    --episodes 100
```

이 명령은 지정한 `model.pt`을 이용하여 SUMO에서 100개 episode를 주행하고, BC 학습에 사용할 transition을 저장하는 방식임

즉, 현재 Pipeline에서는 먼저 주행을 수행하여 Dataset을 만들고, 이후 `train_bc.py`가 이 파일만 읽어서 학습함

```text
model.pt
   ↓
SUMO 주행 100 episodes
   ↓
Observation / Action 기록
   ↓
*.npz 또는 *.jsonl
```

> **중요:** BC는 수집된 행동을 그대로 모방하므로, 어떤 Policy로 데이터를 수집했는지가 최종 BC 성능에 직접적인 영향을 줌

---

# 4. `train_bc.py`가 읽는 데이터

`train_bc.py`는 `.npz` 또는 `.jsonl` 파일을 지원함

## 4.1 NPZ 형식

NPZ에서는 다음 key를 사용함

```text
observation
action_raw
lane_change
episode
```

코드에서는 다음과 같이 읽음

```python
with np.load(path, allow_pickle=False) as data:
    obs, actions, lanes, episodes = [data[k] for k in
        ("observation", "action_raw", "lane_change", "episode")]
```

`detected_xy`와 같은 추가 정보가 저장되어 있어도 BC 학습에는 사용하지 않음

현재 BC가 실제로 사용하는 값은 다음과 같음

```text
Observation  → observation
Acceleration → action_raw[:, 0]
Lane Change  → lane_change
Episode ID   → episode
```

즉, 현재 BC는 **단일 시점의 observation `o_t`만 입력으로 사용**함

Observation history나 privileged state는 사용하지 않음

---

# 5. Dataset shape 및 값 검증

`load_data()`에서는 학습 전에 Dataset이 정상적인지 확인함

Observation은 다음 형태여야 함

```text
(N, obs_dim)
```

Action은 다음 형태여야 함

```text
(N, 2)
```

Lane change와 episode index는 다음 형태여야 함

```text
(N,)
```

또한 다음 조건을 검사함

```text
Observation / Action에 NaN 또는 Inf가 없는지 확인
Raw action 범위가 [-1, 1]인지 확인
Lane change가 {-1, 0, +1} 중 하나인지 확인
```

조건을 만족하지 않으면 학습 전에 `ValueError`가 발생함

따라서 BC 학습이 시작되지 않는 경우에는 먼저 저장된 Dataset의 shape과 action 범위를 확인하면 됨

---

# 6. Lane Change Label 변환

환경에서 저장된 lane change는 다음 값을 사용함

```text
{-1, 0, +1}
```

하지만 `CrossEntropyLoss`는 class index가 `0, 1, 2` 형태여야 하므로 `train_bc.py`에서 다음과 같이 변환함

```python
(1 - lanes).astype(np.int64)
```

따라서 현재 코드 기준 mapping은 다음과 같음

| 저장된 `lane_change` | BC class |
|---:|---:|
| `+1` | `0` |
| `0` | `1` |
| `-1` | `2` |

`algorithms/bc.py`의 `predict()`에서는 다시 역변환하여 환경에서 사용하는 lane command로 반환함

```python
return np.asarray([accel.item(), 1 - lane.item()], dtype=np.float32)
```

즉, **Dataset → 학습 class → 환경 action** 사이의 변환이 코드 내부에서 처리됨

---

# 7. Train / Validation 분리

BC 학습에서는 transition을 무작위로 섞어서 나누는 것이 아니라 **episode 단위로 train / validation을 분리**함

관련 함수는 다음과 같음

```python
def split_episodes(episodes, val_fraction, seed):
    ...
```

기본 validation 비율은 다음과 같음

```text
--val-fraction 0.2
```

예를 들어 100 episodes를 수집했다면 대략

```text
Train      : 80 episodes
Validation : 20 episodes
```

형태로 분리됨

같은 episode의 일부 timestep이 train에 들어가고 나머지가 validation에 들어가는 leakage를 방지하기 위한 구조임

---

# 8. Observation 정규화

BC 모델에 Observation을 그대로 넣지 않고 train data의 평균과 표준편차를 이용하여 정규화함

`train_bc.py`에서 train data만 이용하여 통계값을 계산함

```python
model.obs_mean.copy_(
    torch.as_tensor(obs[train_idx].mean(0), device=device)
)

obs_std = obs[train_idx].std(0)
obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)

model.obs_std.copy_(
    torch.as_tensor(obs_std, device=device)
)
```

Validation data는 평균과 표준편차 계산에 사용하지 않음

따라서 validation information이 training normalization에 섞이지 않도록 구성되어 있음

실제 `BCPolicy.forward()`에서는 다음 계산을 수행함

```python
(obs - self.obs_mean) / self.obs_std
```

`obs_mean`과 `obs_std`는 model buffer로 저장되므로 학습 후 추론에서도 같은 정규화 값을 사용함

---

# 9. `algorithms/bc.py` — BC Policy 구조

현재 BC Policy는 **shared MLP + 2개의 output head** 구조임

```text
Observation
    ↓
Normalization
    ↓
Shared MLP
    ↓
 ┌───────────────┴───────────────┐
 ↓                               ↓
Acceleration Head           Lane Head
1-dimensional output        3-class logits
```

기본 hidden layer 크기는 다음과 같음

```text
256 → 256
```

`BCPolicy`의 핵심 부분은 다음과 같음

```python
self.trunk = build_mlp(
    obs_dim,
    hidden_sizes,
    activation="relu"
)

self.accel_head = nn.Linear(hidden_sizes[-1], 1)
self.lane_head = nn.Linear(hidden_sizes[-1], 3)
```

즉, Observation에서 공통 feature를 추출한 뒤

```text
Acceleration prediction
Lane-change classification
```

을 동시에 수행함

---

# 10. Acceleration 출력

Acceleration은 연속값이므로 regression으로 학습함

BC Policy에서는 acceleration head의 출력에 `tanh`를 적용함

```python
accel = torch.tanh(self.accel_head(z))
```

따라서 model이 예측하는 acceleration raw action은 항상 다음 범위에 들어감

```text
[-1, 1]
```

이는 Dataset 검증 시 사용하는 `action_raw` 범위와 동일함

---

# 11. Lane Change 출력

Lane change는 3-class classification으로 처리함

```python
self.lane_head = nn.Linear(hidden_sizes[-1], 3)
```

forward 결과는 class probability가 아니라 **3개의 logits**임

```python
return accel, self.lane_head(z)
```

Deterministic evaluation에서는 가장 큰 logit을 가지는 class를 선택함

```python
lane = logits.argmax(-1)
```

Stochastic prediction을 사용할 경우에는 categorical distribution에서 sampling할 수 있도록 구현되어 있음

```python
torch.distributions.Categorical(logits=logits).sample()
```

기본 `predict()` 설정은 `deterministic=True`임

---

# 12. BC Loss

`train_bc.py`에서는 acceleration과 lane change에 서로 다른 loss를 사용함

## 12.1 Acceleration loss

Acceleration은 continuous action이므로 MSE를 사용함

```python
mse = F.mse_loss(pred, accel)
```

즉,

```text
Predicted Acceleration
        ↕
Dataset Acceleration
```

차이를 줄이도록 학습함

## 12.2 Lane-change loss

Lane change는 3-class classification이므로 Cross Entropy를 사용함

```python
ce = F.cross_entropy(logits, lane)
```

## 12.3 Total loss

최종 loss는 다음과 같음

```python
loss = mse + lane_weight * ce
```

기본 설정은

```text
--lane-weight 1.0
```

이므로 기본적으로

```text
Total Loss = Acceleration MSE + Lane Cross Entropy
```

형태임

Lane-change loss의 영향을 더 크게 또는 작게 만들고 싶다면 `--lane-weight`를 변경하면 됨

---

# 13. BC 학습 실행

예를 들어 수집한 파일이 `data/pomdp_example.npz`라면 다음과 같이 학습함

```bash
python train_bc.py --data data/pomdp_example.npz
```

기본 학습 설정은 다음과 같음

| Argument | 기본값 | 의미 |
|---|---:|---|
| `--epochs` | `50` | 전체 학습 epoch 수 |
| `--batch-size` | `256` | mini-batch 크기 |
| `--lr` | `3e-4` | Adam learning rate |
| `--hidden-sizes` | `256 256` | MLP hidden layer 크기 |
| `--val-fraction` | `0.2` | validation episode 비율 |
| `--lane-weight` | `1.0` | lane classification loss 가중치 |
| `--seed` | `0` | random seed |
| `--device` | `auto` | 학습 device |

예를 들어 학습 설정을 직접 지정하려면 다음과 같이 실행 가능함

```bash
python train_bc.py \
    --data data/pomdp_example.npz \
    --epochs 100 \
    --batch-size 256 \
    --lr 3e-4 \
    --hidden-sizes 256 256 \
    --lane-weight 1.0
```

---

# 14. Device 설정

현재 `--device`는 다음 옵션을 지원함

```text
auto
cpu
cuda
mps
```

기본값은 `auto`임

현재 코드에서 `auto`는 다음과 같이 동작함

```python
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

따라서 NVIDIA GPU가 있으면 CUDA를 사용하고, 그렇지 않으면 CPU를 사용함

Mac에서 MPS를 사용하려면 현재 코드 기준으로 자동 선택되지 않으므로 다음과 같이 직접 지정하면 됨

```bash
python train_bc.py \
    --data data/pomdp_example.npz \
    --device mps
```

---

# 15. 실제 Training Loop

한 epoch에서 수행되는 과정은 다음과 같음

```text
DataLoader에서 batch 생성
        ↓
Observation을 BCPolicy에 입력
        ↓
Acceleration + Lane logits 예측
        ↓
MSE + Cross Entropy 계산
        ↓
Backpropagation
        ↓
Gradient clipping
        ↓
Adam optimizer update
```

코드에서는 다음과 같이 학습함

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Gradient norm을 `1.0`으로 clipping하여 지나치게 큰 gradient가 발생하는 것을 제한함

---

# 16. 학습 중 출력되는 값

각 epoch마다 train / validation 성능을 계산함

주요 metric은 다음과 같음

```text
loss
accel_mse
lane_ce
lane_accuracy
```

터미널에는 다음과 같은 형태로 출력됨

```text
epoch   1/50 train=... val=... accel_mse=... lane_acc=...
```

확인해야 할 값은 다음과 같음

```text
train loss 감소 여부
validation loss 감소 여부
acceleration MSE 크기
lane-change accuracy
```

단, validation loss가 낮다고 해서 SUMO에서 반드시 안정적으로 주행한다는 의미는 아님

최종적으로 `test.py`를 통한 closed-loop 주행 평가가 필요함

---

# 17. Best Model 저장

Validation loss가 가장 낮아질 때마다 다음 파일을 저장함

```text
model.pt
```

학습 마지막 epoch의 모델도 별도로 저장함

```text
last_model.pt
```

즉,

```text
model.pt
→ validation loss가 가장 낮았던 모델

last_model.pt
→ 마지막 epoch의 모델
```

일반적으로 평가에는 `model.pt`을 사용하는 것이 적절함

---

# 18. 결과 폴더

`--out-dir`을 지정하지 않으면 다음과 같은 형식으로 자동 생성됨

```text
results/
└── bc_YYYYMMDD_HHMMSS_microsecond/
    ├── config.json
    ├── training_log.csv
    ├── model.pt
    └── last_model.pt
```

각 파일의 의미는 다음과 같음

| 파일 | 내용 |
|---|---|
| `config.json` | 학습 hyperparameter, device, obs_dim, train/validation episode 정보 |
| `training_log.csv` | epoch별 train / validation metric |
| `model.pt` | best validation model |
| `last_model.pt` | 마지막 epoch model |

`config.json`에는 실제로 사용된 train episode와 validation episode도 기록됨

따라서 나중에 같은 실험을 확인할 때 유용함

---

# 19. 학습 결과 확인

학습이 끝나면 터미널에 best model 경로와 평가 명령이 출력됨

```text
최적 검증 모델: .../model.pt
주행 평가: python test.py .../model.pt --algorithm bc --nogui
```

예를 들어 결과가 다음 폴더에 저장되었다면

```text
results/bc_20260922_100000_000000/model.pt
```

다음과 같이 평가함

```bash
python test.py \
    results/bc_20260922_100000_000000/model.pt \
    --algorithm bc \
    --nogui
```

GUI로 실제 움직임을 확인하고 싶다면 `--nogui`를 제거하면 됨

---

# 20. `BCPolicy.predict()`가 환경에 Action을 전달하는 과정

평가 시에는 현재 Observation 하나를 입력으로 받아 action을 생성함

```text
Current Observation o_t
        ↓
Observation normalization
        ↓
Shared MLP
        ↓
Acceleration + Lane logits
        ↓
Lane class 선택
        ↓
[accel_raw, lane_cmd]
```

`predict()`의 최종 출력 형식은 다음과 같음

```python
[accel_raw, lane_cmd]
```

따라서 `test.py`와 SUMO Environment는 기존 action interface를 그대로 사용할 수 있음

---

# 21. 현재 BC 모델의 특징

현재 구현의 특징을 정리하면 다음과 같음

```text
입력
→ 단일 시점 Observation o_t

Policy
→ Feed-forward MLP

Acceleration
→ Regression + tanh

Lane Change
→ 3-class Classification

Training
→ Offline supervised learning

Validation
→ Episode 단위 split

Environment interaction during BC training
→ 없음
```

즉, 현재 BC는 **POMDP에서 과거 observation history를 추정하는 모델이 아니라, 현재 observation만으로 action을 예측하는 baseline**임

---

# 22. 자주 확인할 부분

## Q1. Dataset을 읽을 수 없다고 나오는 경우

`.npz` 또는 `.jsonl`인지 확인함

NPZ라면 최소한 다음 key가 있는지 확인함

```text
observation
action_raw
lane_change
episode
```

---

## Q2. Action range 오류가 발생하는 경우

`action_raw` 값이 `[-1, 1]` 범위인지 확인함

Lane change는 반드시 다음 값 중 하나여야 함

```text
-1, 0, +1
```

---

## Q3. Lane accuracy가 지나치게 높은데 실제 차선 변경을 하지 않는 경우

Dataset에서 `lane_change = 0`이 대부분인지 확인해야 함

Keep-lane sample이 지나치게 많으면 높은 accuracy가 나와도 실제 lane-change 학습은 충분하지 않을 수 있음

따라서 class별 sample 수를 함께 확인하는 것이 좋음

---

## Q4. Train loss는 감소하지만 validation loss가 증가하는 경우

Overfitting 가능성이 있음

다음 값을 조절할 수 있음

```text
--epochs
--hidden-sizes
--lr
```

또는 더 다양한 episode를 수집할 필요가 있음

---

## Q5. Validation 성능은 좋은데 SUMO 주행이 불안정한 경우

BC는 저장된 state distribution에서 action을 맞추는 학습임

실제 SUMO에서 model이 작은 실수를 하면 Dataset에서 적게 본 state로 이동할 수 있음

따라서 반드시 closed-loop 주행 결과를 확인해야 함

현재 단계에서는 이론보다 다음 순서로 debugging하는 것이 중요함

```text
Dataset 값 확인
    ↓
Train / Validation metric 확인
    ↓
BC output range 확인
    ↓
SUMO에서 실제 주행 확인
```

---

# 23. 실습 순서 정리

현재 코드 기준 BC 학습 과정은 다음 순서로 진행하면 됨

### 1. 데이터 수집

```bash
python collect_pomdp_data.py \
    --model results/run_20260907_202220/model.pt \
    --episodes 100
```

### 2. 생성된 Dataset 확인

```text
*.npz 또는 *.jsonl
```

### 3. BC 학습

```bash
python train_bc.py --data data/pomdp_example.npz
```

### 4. Training log 확인

```text
training_log.csv
```

### 5. Best model 확인

```text
model.pt
```

### 6. SUMO에서 평가

```bash
python test.py results/.../model.pt --algorithm bc --nogui
```

---

# 24. 코드 수정 시 주로 볼 위치

| 수정하고 싶은 내용 | 확인할 위치 |
|---|---|
| 수집 episode 수 | `collect_pomdp_data.py` 실행 시 `--episodes` |
| BC 입력 Observation | Dataset의 `observation` 및 Environment observation 구성 |
| MLP 크기 | `train_bc.py --hidden-sizes` |
| Network output 구조 | `algorithms/bc.py` |
| Learning rate | `train_bc.py --lr` |
| Batch size | `train_bc.py --batch-size` |
| Lane loss 중요도 | `train_bc.py --lane-weight` |
| Validation 비율 | `train_bc.py --val-fraction` |
| 학습 epoch | `train_bc.py --epochs` |
| CPU / GPU / MPS | `train_bc.py --device` |
| 저장 위치 | `train_bc.py --out-dir` |
| 실제 주행 성능 | `test.py` |

---

# 25. 이번 단계에서 확인할 것

- [ ] `collect_pomdp_data.py`로 충분한 episode가 수집되는가?
- [ ] NPZ/JSONL에 필요한 key가 정상적으로 저장되는가?
- [ ] Observation shape이 모든 sample에서 동일한가?
- [ ] `action_raw`이 `[-1, 1]` 범위인가?
- [ ] `lane_change`가 `{-1, 0, +1}`인가?
- [ ] Train / Validation이 episode 단위로 분리되는가?
- [ ] Train loss와 validation loss가 감소하는가?
- [ ] Lane-change class가 지나치게 불균형하지 않은가?
- [ ] Best model인 `model.pt`이 정상적으로 저장되는가?
- [ ] `test.py`에서 BC Policy가 실제 SUMO 환경을 주행하는가?

---

# Next

BC는 수집된 행동을 모방하는 baseline임

다음 단계에서는 같은 SUMO Environment에서 Reward를 이용하여 Policy를 직접 개선하는 Reinforcement Learning을 적용할 수 있음

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
