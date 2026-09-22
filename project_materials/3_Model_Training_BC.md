# 3. Model Training — Behavior Cloning

> **목표:** 저장한 관측과 행동으로 BC 모델을 학습하고, 모델 저장 및 성능 평가 실습

## 1. 전체 흐름

```text
학습된 모델 불러오기
        ↓
collect_pomdp_data.py: SUMO 주행
        ↓
NPZ / JSONL 데이터 + 수집 메타데이터
        ↓
train_bc.py: observation → action_raw 지도학습
        ↓
algorithms/bc.py의 BCPolicy
        ↓
model.pt 저장
        ↓
BCPolicy.load()로 복원
        ↓
BC 평가 연결 후 SUMO 주행 평가
```

이번 시간 실습은 강화학습이 아닌, 수집 정책의 행동을 모방하는 Imitation Learning임

데이터 수집과 주행 평가에는 SUMO가 필요하지만, BC 학습 자체는 NumPy와 PyTorch로 저장된 데이터만 읽으며 SUMO를 실행하지 않음

| 파일 | 역할 |
|---|---|
| `collect_pomdp_data.py` | 학습된 모델로 SUMO 주행, transition 저장 |
| `train_bc.py` | 데이터 검증, 모델 학습 |
| `algorithms/bc.py` | `BCPolicy` 신경망, 추론, 모델 저장· 모델 불러오기 |
| `test.py` | 현재 모델의 SUMO 주행 평가 |
| `env/sumo_env.py` | raw 행동 제한, 차선 명령 양자화, 안전 조건에 따른 실행 |

## 2. 실행 준비

아래 명령은 프로젝트 최상위 폴더에서 실행한다. 만약 `sumo-rl` Conda 환경을 사용하는 경우 먼저 활성화한다.

```bash
conda activate sumo-rl
```

## 3. 데이터 수집과 기존 데이터 선택

현재 데이터 수집 시 학습된 모델을 읽는 `--model` 옵션을 추가하여 데이터 수집이 가능함.

```bash
python collect_pomdp_data.py --model results/run_20260907_202220/model.pt --episodes 10
```

화면을 보며 수집하려면 다음과 같이 옵션 추가

```bash
python collect_pomdp_data.py --model results/run_20260907_202220/model.pt --episodes 10 --gui
```

랜덤 정책은 가감속을 `[-1, 1]`에서 균등하게 뽑고, 차선 명령을 오른쪽 15%, 유지 70%, 왼쪽 15%로 선택한다. 차선 유지 정책은 `[0.25, 0.0]`을 반환함. 

수집 결과는 `data/bc_demo.npz`와 `data/bc_demo.jsonl`임. 이름을 생략하면 날짜·시각이 사용됨. 현재 코드는 같은 이름의 파일을 덮어쓸 수 있으므로 재수집 시 새 이름을 사용함.

이미 모델로 수집한 다음 데이터가 있다면, 현재 수집기를 다시 실행하지 않고 그대로 학습 진행이 가능함.

```bash
python train_bc.py --data data/pomdp_20260922_095037_217175.npz --epochs 50
```

BC는 저장된 ego 행동을 모방한다. 배경차가 IDM을 사용한다고 해서 ego 행동이 IDM 전문가 행동이 되는 것은 아니다.

## 4. BC 학습에 사용하는 데이터

NPZ와 JSONL 모두 다음 세 필드만 사용한다.

| 필드 | 배열 형태 | 용도 |
|---|---|---|
| `observation` | `(N, state_dim)` | 현재 시점의 부분관측 |
| `action_raw` | `(N, 2)` | 정답 가감속·차선변경 raw 값 |
| `episode` | `(N,)` | 학습·검증 분리용 에피소드 ID |

`N`은 전체 transition 수이다. `state_dim`은 데이터에서 읽으며, 현재 환경의 관측은 31차원이다. 코드의 `state`는 SUMO 전체 내부 상태가 아니라 에이전트가 받는 부분관측을 의미한다.

NPZ에서는 다음과 같이 읽는다.

```python
with np.load(path, allow_pickle=False) as data:
    obs = data["observation"]
    actions = data["action_raw"]
    episodes = data["episode"]
```

JSONL에서는 각 줄의 동일한 필드를 읽는다. `lane_change`, reward, next observation, privileged state, 감지 차량 좌표는 현재 BC 학습에 사용하지 않는다. 따라서 NPZ의 object 배열인 `detected_xy`를 읽기 위해 pickle을 허용할 필요도 없다.

### 행동 값의 의미

```python
action_raw = [accel_raw, lane_change_raw]
```

가감속 raw 값은 물리 단위의 가속도 자체가 아니라 환경에 전달하는 제어 입력이다. 학습 정답은 두 축 모두 `[-1, 1]` 범위이다.

현재 수집기는 생성한 정책 행동을 환경에 전달하고 그 값을 `action_raw`로 저장한다. 별도의 `lane_change` 필드는 양자화된 요청 명령이며, 실제 차선변경 성공 여부와는 다르다. JSONL의 `action.lane_change_applied`로 실행 여부를 확인할 수 있다.

### 학습 전 검증

`load_data()`는 다음 조건을 확인한다.

- 데이터가 비어 있지 않고 관측이 `(N, state_dim)` 형태인지
- 행동이 `(N, 2)`, 에피소드 ID가 `(N,)`인지
- 세 배열의 sample 수가 같은지
- 관측과 행동에 NaN 또는 Inf가 없는지
- 에피소드 ID가 유한한 정수 값인지
- 두 raw 행동 값이 모두 `[-1, 1]` 범위인지

차선 raw 값은 연속값이므로 반드시 `-1`, `0`, `+1` 중 하나일 필요는 없다. 현재 코드에는 차선 class index 변환이 없다.

## 5. 학습·검증 분리와 정규화

`split_episodes()`는 에피소드 ID를 seed에 따라 섞은 뒤 검증 에피소드를 선택한다. 같은 에피소드의 transition이 학습과 검증에 함께 들어가지 않는다.

기본 `--val-fraction 0.2`에서 100개 에피소드는 학습 80개, 검증 20개로 나뉜다. 에피소드 길이는 서로 다르므로 transition 수가 정확히 80:20이 되는 것은 아니다. 최소 2개 에피소드가 필요하다.

정규화 통계는 학습 관측에서만 계산한다.

```python
obs_mean = obs[train_idx].mean(axis=0)
obs_std = obs[train_idx].std(axis=0)
obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
model.obs_mean.copy_(torch.as_tensor(obs_mean, device=device))
model.obs_std.copy_(torch.as_tensor(obs_std, device=device))
```

학습에서 값이 일정한 항목은 표준편차를 1로 둔다. 매우 작은 값으로 나누어 검증·추론 입력이 폭증하는 것을 방지하기 위한 처리이다.

모델 내부에서는 다음 계산을 수행한다.

```python
state = (state - self.obs_mean) / self.obs_std
```

평균과 표준편차는 `register_buffer()`로 등록되어 가중치와 함께 저장된다. 검증과 주행 추론에서도 동일한 통계를 사용한다.

## 6. BCPolicy 구조

```text
부분관측 (state_dim)
        ↓
정규화
        ↓
Linear(state_dim, 256) + ReLU
        ↓
Linear(256, 256) + ReLU
        ↓
Linear(256, 2)
        ↓
[accel_raw, lane_change_raw]
```

현재 `algorithms/bc.py`의 구현은 다음과 같다.

```python
self.fc1 = nn.Linear(state_dim, 256)
self.fc2 = nn.Linear(256, 256)
self.fc3 = nn.Linear(256, action_dim)
```

```python
def forward(self, state):
    state = (state - self.obs_mean) / self.obs_std
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)
```

가감속과 차선변경을 하나의 출력층에서 모두 회귀한다. 별도의 lane classification head, logits, argmax 변환은 없다. 출력층에는 tanh나 clipping이 없으므로 모델 출력 자체가 `[-1, 1]`에 제한되지는 않는다.

단일 시점의 관측만 사용하며, 과거 관측을 기억하는 RNN이나 belief 추정 구조는 없다. POMDP 환경에서 사용하는 feed-forward BC baseline이다.

## 7. 손실과 학습 과정

두 행동 값 전체에 MSE를 적용한다.

```python
pred_action = model(state)
loss = F.mse_loss(pred_action, target_action)
```

배치 크기가 `B`이면 손실은 `B × 2`개 원소의 제곱오차 평균이다. 가감속과 차선변경에 동일한 가중치를 사용한다. Cross Entropy나 별도의 차선 손실 가중치는 없다.

학습 배치에서는 다음을 실행한다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

최적화기는 Adam이며 현재 학습 코드에는 gradient clipping이 없다. 검증에서는 gradient 계산과 가중치 갱신을 하지 않는다. 에피소드 분리 후 학습 DataLoader는 sample을 섞고 검증 DataLoader는 섞지 않는다.

## 8. 학습 실행과 옵션

앞에서 생성한 데이터로 학습한다.

```bash
python train_bc.py --data data/bc_demo.npz --epochs 50
```

같은 수집 결과의 JSONL을 지정해도 된다.

```bash
python train_bc.py --data data/bc_demo.jsonl --epochs 50
```

수집이 끝난 파일을 지정한다. 이미 데이터가 있다면 재수집 없이 `--data`에 그 파일 경로를 넣는다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--data` | 필수 | NPZ 또는 JSONL 파일 경로 |
| `--epochs` | `50` | 전체 학습 반복 횟수 |
| `--batch-size` | `256` | 배치 크기 |
| `--lr` | `3e-4` | Adam 학습률 |
| `--val-fraction` | `0.2` | 검증 에피소드 비율 |
| `--seed` | `0` | 데이터 분리와 난수 생성 seed |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `mps` |
| `--out-dir` | 자동 생성 | 새 결과 폴더 경로 |

은닉층 크기는 `algorithms/bc.py`에 `256 → 256`으로 고정되어 있다. `--hidden-sizes`와 `--lane-weight` 옵션은 현재 지원하지 않는다.

`auto`는 **CUDA → MPS → CPU** 순서로 사용 가능한 장치를 선택한다. Apple Silicon의 MPS도 지원 여부에 따라 자동 선택된다. CPU를 지정하려면 다음과 같이 실행한다.

```bash
python train_bc.py --data data/bc_demo.npz --epochs 50 --batch-size 256 --lr 3e-4 --device cpu --out-dir results/bc_demo
```

이후 평가 예시는 위 명령의 `results/bc_demo`를 기준으로 한다. 결과 폴더는 새로 생성하며 이미 존재하면 중단한다. 재실행할 때는 다른 `--out-dir`을 지정하거나 생략해 자동 이름을 사용한다.

## 9. 로그와 저장 결과

학습 중 출력 형식은 다음과 같다. 숫자는 실행마다 달라진다.

```text
Epoch   1/50 | train_loss=0.149120 | val_loss=0.067159
```

두 값 모두 전체 행동 벡터의 MSE이다. 현재는 별도의 가감속 MSE, lane CE, lane accuracy를 출력하지 않는다.

```text
results/bc_demo/
├── config.json
├── training_log.csv
├── model.pt
└── last_model.pt
```

| 파일 | 내용 |
|---|---|
| `config.json` | 데이터 경로, 학습 설정, device, `state_dim`, `action_dim`, sample 수, 분할 에피소드 ID |
| `training_log.csv` | `epoch`, `train_loss`, `val_loss` |
| `model.pt` | 검증 MSE가 가장 낮았던 모델 |
| `last_model.pt` | 마지막 epoch 모델 |

`--out-dir`을 생략하면 `results/bc_YYYYMMDD_HHMMSS_microsecond/` 형태로 생성한다.

`model.pt`는 raw state_dict만 담는 파일이 아니다. `BCPolicy.save()`는 `algorithm`, `state_dim`, `action_dim`, `state_dict`를 저장한다. 정규화 통계도 state_dict에 포함된다. optimizer 상태와 학습 재개 옵션은 제공하지 않는다.

학습 종료 시 `Best model`, `Last model`, `Training log`, `Evaluation` 항목이 출력된다. 기본적으로 평가에는 `model.pt`를 사용한다. 단, 출력되는 `Evaluation` 명령의 `--algorithm bc`는 현재 `test.py`에서 지원하지 않으므로 그대로 실행할 수 없다.

## 10. 모델 복원과 행동 생성

```python
from algorithms.bc import BCPolicy

policy = BCPolicy.load("results/bc_demo/model.pt", device="cpu")
# obs는 환경에서 받은 단일 관측으로 shape이 (policy.state_dim,)이어야 한다.
# action = policy.predict(obs)
```

`predict()`는 단일 관측을 float32 텐서로 바꾸고 모델과 같은 장치로 옮긴 뒤, 정규화와 신경망 계산을 수행한다. 결과는 길이 2의 NumPy 배열이다. `deterministic` 인자는 평가 인터페이스 호환용이며, 현재 BC는 False여도 확률적으로 샘플링하지 않는다.

```text
관측 → 정규화 → MLP → raw 행동 2개
        ↓
환경에서 [-1, 1]로 제한
        ↓
차선 raw 값을 {-1, 0, +1} 명령으로 양자화
        ↓
차선 범위·안전 조건에 따라 실행
```

현재 `ACTION["lane_change_threshold"]`는 `0.5`이다.

| 차선 raw 값 | 요청 명령 |
|---|---|
| `>= 0.5` | 왼쪽 `+1` |
| `<= -0.5` | 오른쪽 `-1` |
| 그 사이 | 유지 `0` |

예를 들어 차선 정답이 `+1`인데 예측이 `0.4`이면 오차는 줄었더라도 환경은 차선을 유지한다. 따라서 낮은 MSE만으로 차선변경 동작을 판단할 수 없다.

## 11. SUMO 평가 연결 상태

현재 `test.py`는 `PPO` 객체를 생성하고 PPO 가중치를 로드한다. `--algorithm` 옵션과 `BCPolicy.load()` 분기가 없으므로, BC 모델을 위치 인자로 넘기는 것만으로는 평가할 수 없다. `train_bc.py`가 출력하는 `--algorithm bc` 평가 명령도 현재는 인자 오류가 발생한다.

현재 실행 가능한 다음 명령은 **PPO 모델 평가**이다.

```bash
python test.py results/run_20260907_202220/model.pt --episodes 5 --nogui
```

BC 주행 평가를 연결할 때는 다음 순서가 필요하다.

1. `BCPolicy.load(model_path)`로 모델을 복원한다.
2. `state_dim`, `action_dim`이 환경의 관측·행동 차원과 맞는지 확인한다.
3. `utils.evaluator.evaluate_policy(policy, env, n_episodes=5, deterministic=True)`에 BC 정책을 전달한다.
4. 평가 종료 시 `env.close()`를 호출한다.

`BCPolicy.predict()`는 평가기가 요구하는 인터페이스를 갖추고 있다. 평가기의 차선변경 지표 이름은 `lane_changes`인데, 현재 `test.py`의 출력부는 `ep_lane_changes`를 참조하므로 이 부분도 수정해야 한다.

주행 평가에서는 충돌률, 완주율, 속도, 차간거리와 차선변경 횟수를 확인한다. 환경은 현재 `env/` 설정을 사용하며 BC 체크포인트가 수집 당시 환경을 자동 복원하지 않는다. 관측 차원뿐 아니라 항목 순서와 의미도 수집 당시와 일치해야 한다.

## 12. 결과 해석과 오류 점검

| 상황 | 확인할 내용 |
|---|---|
| 데이터 로딩 실패 | 파일 경로, NPZ/JSONL 형식, 필수 필드 3개 |
| raw 행동 범위 오류 | `action_raw` 두 축이 모두 `[-1, 1]`인지 |
| 에피소드 분리 실패 | 서로 다른 에피소드가 최소 2개 있는지 |
| 인식하지 못하는 CLI 옵션 | `python train_bc.py --help`에 있는 옵션인지 |
| 결과 폴더 생성 실패 | `--out-dir`이 이미 존재하는지 |
| 모델 관측·행동 차원 오류 | 수집·학습·평가의 관측/행동 구성이 같은지 |
| 이전 모델 로드 실패 | 현재 `BCPolicy.save()`로 저장한 회귀 모델인지. 이전 분류 모델 체크포인트는 구조가 다름 |
| 검증 MSE만 증가 | 과적합 또는 데이터 분포 차이. 학습률·epoch와 수집 에피소드 다양성 확인 |
| MSE는 낮지만 차선변경이 적음 | 정답 행동 분포와 raw 예측이 임계값을 넘는지 확인 |
| 주행 중 성능 저하 | 작은 행동 오차로 수집 데이터에서 드문 관측에 진입할 수 있으므로 실제 주행 평가 확인 |

BC는 수집 정책의 좋은 행동과 잘못된 행동을 함께 모방한다. 보상을 저장하더라도 현재 학습에서는 사용하지 않는다. 차선 유지 데이터가 대부분이거나 같은 관측에 상반된 명령이 많으면, MSE 회귀 결과가 유지에 가까운 평균값으로 모일 수 있다.

## 13. 코드 변경 위치

| 변경할 내용 | 위치 |
|---|---|
| 수집 정책·에피소드 수 | `collect_pomdp_data.py --policy ... --episodes ...` |
| 학습 모델 기반 수집 | 수집기에 모델 로드와 `predict()` 연결 필요 |
| 학습률·배치 크기·epoch | `train_bc.py --lr`, `--batch-size`, `--epochs` |
| 검증 비율·seed | `train_bc.py --val-fraction`, `--seed` |
| CPU/GPU 선택 | `train_bc.py --device` |
| 정규화 통계 계산 | `train_bc.py`의 모델 생성 직후 |
| 은닉층 크기·출력 구조 | `algorithms/bc.py`의 `BCPolicy` |
| 손실 함수 | `train_bc.py`의 `run_epoch()` |
| 차선 양자화 임계값 | `env/mdp_config.py`의 `ACTION` |
| 주행 평가 | `test.py`, `utils/evaluator.py` |

## 14. 실습 확인 사항

- [ ] 데이터가 모델 기반 주행인지 랜덤/차선 유지 주행인지 확인했는가?
- [ ] 기존 데이터에 메타데이터가 있다면 수집 정책을 확인했는가?
- [ ] 필수 데이터의 shape과 raw 행동 범위가 올바른가?
- [ ] 학습·검증이 에피소드 단위로 분리되는가?
- [ ] 정규화 통계가 학습 데이터에서만 계산되는가?
- [ ] 학습·검증 MSE와 best model 저장을 확인했는가?
- [ ] BC 평가 연결 후 SUMO에서 충돌·완주·차선변경을 확인했는가?

다음 강화학습 단계에서는 수집 행동을 정답으로 맞추는 대신, 환경과 상호작용하며 보상을 이용해 정책을 개선한다.
