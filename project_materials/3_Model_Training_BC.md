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
source ~/miniforge3/etc/profile.d/conda.sh
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

랜덤 정책은 가감속을 `[-1, 1]`에서 균등하게 뽑고, 차선 명령을 오른쪽 15%, 유지 70%, 왼쪽 15%로 선택하고, 차선 유지 정책은 `[0.25, 0.0]`을 반환함. 

수집 결과는 `data/bc_demo.npz`와 `data/bc_demo.jsonl`임. 이름을 생략하면 날짜·시각이 사용됨. 현재 코드는 같은 이름의 파일을 덮어쓸 수 있으므로 재수집 시 새 이름을 사용함.

이미 모델로 수집한 다음 데이터가 있다면, 현재 수집기를 다시 실행하지 않고 그대로 학습 진행이 가능함.

```bash
python train_bc.py --data data/pomdp_20260922_095037_217175.npz --epochs 50
```

BC는 저장된 ego 행동을 모방하며, 주변 차량을 IDM으로 사용한다고 해서 ego 행동이 IDM 전문가 행동이 되는 것이 아님. 우리가 수집한 데이터는 기존에 학습되어 있는 자율주행 차량의 주행 데이터이기 때문.

## 4. BC 학습에 사용하는 데이터

NPZ와 JSONL 모두 다음 세 필드만 사용함.

| 필드 | 배열 형태 | 용도 |
|---|---|---|
| `observation` | `(N, state_dim)` | 현재 시점의 부분관측 |
| `action_raw` | `(N, 2)` | 정답 가감속·차선변경 raw 값 |
| `episode` | `(N,)` | 학습·검증 분리용 에피소드 ID |

`N`은 전체 transition 수 

`state_dim`은 데이터에서 읽으며, 현재 환경의 관측은 31차원임

코드의 `state`는 SUMO 전체 내부 상태가 아니라 에이전트가 받는 부분관측을 의미함.

NPZ에서는 다음과 같이 읽음

```python
with np.load(path, allow_pickle=False) as data:
    obs = data["observation"]
    actions = data["action_raw"]
    episodes = data["episode"]
```

JSONL에서는 각 줄의 동일한 필드를 읽음

`lane_change`, reward, next observation, privileged state, 감지 차량 좌표는 현재 BC 학습에 사용하지 않음. 

### 행동 값의 의미

```python
action_raw = [accel_raw, lane_change_raw]
```

가감속 raw 값은 물리 단위의 가속도 자체가 아니라 환경에 전달하는 제어 입력으로, 학습 정답은 두 축 모두 `[-1, 1]` 범위임.

현재 수집기는 생성한 정책 행동을 환경에 전달하고 그 값을 `action_raw`로 저장함. 별도의 `lane_change` 필드는 양자화된 명령이며, 실제 차선 변경 성공 여부와는 다름. JSONL의 `action.lane_change_applied`로 실행 여부를 확인할 수 있음.

### 학습 전 검증

`load_data()`는 다음 조건을 확인함.

- 데이터가 비어 있지 않고 관측이 `(N, state_dim)` 형태인지
- 행동이 `(N, 2)`, 에피소드 ID가 `(N,)`인지
- 세 배열의 sample 수가 같은지
- 관측과 행동에 NaN 또는 Inf가 없는지
- 에피소드 ID가 유한한 정수 값인지
- 두 raw 행동 값이 모두 `[-1, 1]` 범위인지

## 5. BCPolicy 구조

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

현재 `algorithms/bc.py`의 구현은 다음과 같음.

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

단일 시점의 관측만 사용하며, 과거 관측을 기억하는 RNN이나 belief 추정 구조는 없음. POMDP 환경에서 사용하는 feed-forward BC baseline임

## 7. 손실과 학습 과정

두 행동 값 전체에 MSE를 적용함.

```python
pred_action = model(state)
loss = F.mse_loss(pred_action, target_action)
```

학습 배치에서는 다음을 실행함.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 8. 학습 실행과 옵션

앞에서 생성한 데이터로 학습함.

```bash
python train_bc.py --data data/bc_demo.npz --epochs 50
```

같은 수집 결과의 JSONL을 지정해도 된다.

```bash
python train_bc.py --data data/bc_demo.jsonl --epochs 50
```

이미 데이터가 있다면 `--data` 옵션에 그 파일 경로를 넣는다.

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

은닉층 크기는 `algorithms/bc.py`에 `256 → 256`으로 고정되어 있음. `--hidden-sizes`와 `--lane-weight` 옵션은 현재 지원하지 않는다.

`auto`는 **CUDA → MPS → CPU** 순서로 사용 가능한 장치를 선택하게 되며, Apple Silicon의 MPS도 지원 여부에 따라 자동 선택됨. CPU를 지정하려면 다음과 같이 실행하면 됨.

```bash
python train_bc.py --data data/bc_demo.npz --epochs 50 --batch-size 256 --lr 3e-4 --device cpu --out-dir results/bc_demo
```

이후 평가 예시는 위 명령의 `results/bc_demo`를 기준으로 함. 결과 폴더는 새로 생성하며 이미 존재하면 중단한다. 재실행할 때는 다른 `--out-dir`을 지정하거나 생략해 자동 이름을 사용함.

## 9. 로그와 저장 결과

학습 중 출력 형식은 다음과 같으며, 숫자는 실행마다 달라짐.

```text
Epoch   1/50 | train_loss=0.149120 | val_loss=0.067159
```

두 값 모두 전체 행동 벡터의 MSE임. 현재는 별도의 가감속 MSE, lane CE, lane accuracy를 출력하지 않음.

```text
results/bc_demo/
├── config.json
├── training_log.csv
├── model.pt
├── last_model.pt
└── tensorboard/
    └── events.out.tfevents.*
```

| 파일 | 내용 |
|---|---|
| `config.json` | 데이터 경로, 학습 설정, device, `state_dim`, `action_dim`, sample 수, 분할 에피소드 ID |
| `training_log.csv` | `epoch`, `train_loss`, `val_loss` |
| `model.pt` | 검증 MSE가 가장 낮았던 모델 |
| `last_model.pt` | 마지막 epoch 모델 |
| `tensorboard/` | TensorBoard 이벤트 로그 |

`--out-dir`을 생략하면 `results/bc_YYYYMMDD_HHMMSS/` 형태로 생성함.
예를 들어 `results/bc_20260922_143025/model.pt`처럼 날짜와 시각(초)까지만 표시하며,
뒤에 마이크로초 숫자는 붙이지 않음. 모델 파일명은 `model.pt`로 유지됨.
같은 초에 시작하여 결과 폴더 이름이 겹치면 기존 결과를 덮어쓰지 않고 중단하므로,
잠시 후 다시 실행하거나 새로운 `--out-dir`을 지정하면 됨.

학습 종료 시 `Best model`, `Last model`, `Training log`, `Evaluation` 항목이 출력됨. 

기본적으로 평가에는 `model.pt`를 사용

### 9.1 TensorBoard에서 학습 곡선 보기

현재 `train_bc.py`는 매 epoch의 결과를 `결과 폴더/tensorboard/`에 자동 기록함.
학습을 실행한 상태에서 다른 터미널을 열고, 프로젝트 최상위 폴더에서 다음 명령을 실행함.

```bash
conda activate sumo-rl
tensorboard --logdir results
```

브라우저에서 **http://localhost:6006**으로 접속하고 Scalars 화면에서 확인함.
여러 학습 결과를 비교하려면 표시할 run을 선택하면 됨.

| 항목 | 의미 |
|---|---|
| `Loss/train` | 해당 epoch의 학습 MSE |
| `Loss/validation` | 해당 epoch의 검증 MSE |
| `Loss/best_validation` | 해당 epoch까지의 최저 검증 MSE |
| `Optimization/learning_rate` | Adam 학습률 |

x축은 epoch이며, 학습·검증 loss는 CSV에 기록된 값과 동일함.
학습 설정은 Text 화면의 `config` 항목에 기록됨.
현재 학습률 스케줄러가 없으므로 학습률은 지정한 `--lr` 값으로 일정함.

특정 실험만 보려면 해당 폴더를 지정함.

```bash
tensorboard --logdir results/bc_demo/tensorboard
```

TensorBoard가 설치되어 있지 않으면 학습에 사용하는 Python 환경에서 설치함.

```bash
python -m pip install tensorboard
```

로그는 매 epoch마다 디스크에 반영하며, 정상 종료 또는 Ctrl+C·예외 발생 시 writer를 닫음.
TensorBoard 기록 기능을 추가한 이후 새로 실행한 학습부터 확인할 수 있고,
기존 CSV만 있는 과거 결과가 자동으로 TensorBoard 로그로 변환되지는 않음.

## 10. `test.py`로 BC 모델 주행 평가

현재 `test.py`는 **BC와 PPO 모델을 모두 지원**함. `train_bc.py`에서 저장한 `model.pt`를 불러와 SUMO에서 주행하고 지표를 출력하게 됨.

### 10.1 화면 없이 BC 평가

프로젝트 최상위 폴더에서 다음과 같이 실행함. `results/bc_demo/model.pt`는 실제 학습 결과의 모델 경로로 바꿔야 함.

```bash
python test.py results/bc_demo/model.pt --algorithm bc --episodes 5 --nogui
```

`model.pt`는 검증 MSE가 가장 낮았던 모델이고, `last_model.pt`는 마지막 epoch 모델임. 마지막 모델을 평가하려면 파일 경로만 바꾸면 됨.

### 10.2 GUI로 주행 확인

`--nogui`를 생략하면 SUMO GUI가 열림.

```bash
python test.py results/bc_demo/model.pt --algorithm bc --episodes 3
```

각 에피소드에서 GUI의 ▶(플레이) 버튼을 눌러 주행을 시작함. 에피소드가 끝나고 다음 창이 열리면 다시 ▶ 버튼을 누른다.

### 10.3 실행 옵션과 자동 판별

| 인자 / 옵션 | 기본값 | 의미 |
|---|---|---|
| `model` | 최근 모델 | 평가할 모델 파일 경로 |
| `--algorithm` | `auto` | `auto`, `bc`, `ppo` 중 선택 |
| `--episodes` | `3` | 평가 에피소드 수. 1 이상 |
| `--nogui` | 미지정 | 지정하면 화면 없이 평가 |

기본값 `auto`는 체크포인트 형식으로 BC/PPO를 판별하며, 따라서 다음 명령도 BC 모델을 자동으로 불러옴.

```bash
python test.py results/bc_demo/model.pt --episodes 5 --nogui
```

`--algorithm bc`를 명시했는데 파일이 PPO 모델이면 오류가 남. 모델 경로를 생략하면 `results/*/model.pt` 중 가장 최근에 수정된 파일을 선택하게 되며, 이때 BC 모델만 검색하는 것은 아니므로, 특정 실험을 평가할 때는 경로를 직접 지정하는 게 좋음.

기존 PPO 모델도 평가할 수 있음.

```bash
python test.py results/run_20260907_202220/model.pt --algorithm ppo --episodes 5 --nogui
```

### 10.4 BC 모델 로드와 평가 과정

```text
체크포인트 유형 판별
        ↓
BC 모델의 state_dim / action_dim과 환경 차원 비교
        ↓
BCPolicy.load(): 가중치와 정규화 통계 복원
        ↓
현재 관측 → policy.predict() → raw 행동 2개
        ↓
SUMO 환경에서 행동 제한·차선 양자화·주행
        ↓
에피소드별 결과 집계 → 평균 지표 출력
```

현재 BC는 CPU로 로드하며 `deterministic=True`로 평가함. 평가에는 `utils/evaluator.py`의 `evaluate_policy()`를 사용하며, 모델 로드 또는 평가 중 오류가 나더라도 `finally`에서 환경을 종료하게 됨.

### 10.5 결과 확인

터미널에는 평가 알고리즘과 다음 주행 지표가 출력됨.

- 충돌률, 완주율, 시간초과율
- 평균 속도, 평균·최소 차간거리
- 에피소드당 차선변경 횟수
- 평균 누적 보상, 평균 에피소드 길이

차선변경 횟수는 실제 실행된 변경을 집계한 `metrics.lane_changes`를 사용함. 검증 MSE가 낮더라도 실제 주행에서 충돌하거나 차선변경이 부족할 수 있으므로, 손실과 주행 지표를 함께 확인해야 함.

BC 체크포인트가 수집 당시 환경을 자동으로 복원하지는 않으므로, 관측 차원뿐 아니라 항목의 순서와 의미도 수집 당시와 일치해야 한다.

## 11. 결과 해석과 오류 점검

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

BC는 수집 정책의 좋은 행동과 잘못된 행동을 함께 모방함. 보상을 저장하더라도 현재 학습에서는 사용하지 않음.

## 12. 코드 변경 위치

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

## 13. 실습 확인 사항

- [ ] 데이터가 모델 기반 주행인지 랜덤/차선 유지 주행인지 확인했는가?
- [ ] 기존 데이터에 메타데이터가 있다면 수집 정책을 확인했는가?
- [ ] 필수 데이터의 shape과 raw 행동 범위가 올바른가?
- [ ] 학습·검증이 에피소드 단위로 분리되는가?
- [ ] 정규화 통계가 학습 데이터에서만 계산되는가?
- [ ] 학습·검증 MSE와 best model 저장을 확인했는가?
- [ ] BC 평가 연결 후 SUMO에서 충돌·완주·차선변경을 확인했는가?

다음 강화학습 단계에서는 수집 행동을 정답으로 맞추는 대신, 환경과 상호작용하며 보상을 이용해 정책을 개선함.
