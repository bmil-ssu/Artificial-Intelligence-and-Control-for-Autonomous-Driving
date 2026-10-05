# 4. Model Training — Reinforcement Learning

> **목표:** `train.py`로 SUMO 주행 경험을 수집하고, `algorithms/ppo.py`의 PPO로 모델을 학습한 뒤 저장·평가 실습

## 1. 전체 흐름과 파일 위치

BC는 저장된 관측과 정답 행동을 비교하지만, 이번 단계에서는 차량이 SUMO에서 직접 행동하고 받은 reward로 모델을 업데이트함. 별도의 NPZ 데이터 파일을 준비하지 않아도 됨.

```text
train.py: 학습 설정 및 실행
        ↓
env/road_builder.py: SUMO 도로 생성
        ↓
env/sumo_env.py: 관측 → 행동 적용 → reward 반환
        ↓
algorithms/ppo.py: 경험 수집 및 모델 업데이트
        ↓
results/run_날짜_시각/: 모델·로그·설정 저장
        ↓
test.py: 저장한 모델로 주행 평가
```

| 파일 경로 | 역할 |
|---|---|
| `train.py` | 학습 설정, 환경·에이전트 생성, 학습 실행 및 저장 |
| `algorithms/ppo.py` | `PPO` 클래스: 행동 선택, 경험 수집, loss 계산, 모델 업데이트 |
| `utils/networks.py` | 정책 및 가치 신경망인 `HybridActorCritic` |
| `utils/buffer.py` | `RolloutBuffer`: 이번 업데이트에 사용할 주행 기록 관리 |
| `env/sumo_env.py` | SUMO 연결, `reset()`, `step()`, 관측·보상 계산 |
| `env/road_config.py` | `ROAD`, `EGO`, `TRAFFIC`: 도로·차량·교통 설정 |
| `env/mdp_config.py` | 관측·행동·시뮬레이션·기본 reward 설정 |
| `utils/logger.py` | `RunLogger`: 콘솔·CSV·TensorBoard 기록 |
| `utils/evaluator.py` | `evaluate_policy()`: 여러 에피소드의 주행 성능 측정 |
| `test.py` | 저장한 PPO/BC 모델로 주행 평가 |

## 2. 실행 준비와 학습 시작

프로젝트를 새로 내려받는 경우 다음 명령을 실행함.

```bash
git clone https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving
```

Mac에서 `sumo-rl` Conda 환경을 사용하는 경우 먼저 활성화함.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sumo-rl
```

프로젝트 최상위 폴더에서 실행함.

```bash
python train.py
```

<img width="400" alt="학습 실행 화면" src="https://github.com/user-attachments/assets/6331be7b-8590-4729-b74c-2edff8be8b6e" />

현재 학습 환경은 `gui=False`이므로 SUMO 창이 뜨지 않음. 주행 화면은 학습 후 `test.py`로 확인함.

## 3. train.py에서 학습 설정 변경하기

BC는 `train_bc.py`의 실행 옵션으로 설정하지만, 현재 RL 학습은 `train.py` 상단의 변수와 `HPARAMS`를 수정함. 다음은 현재 설정 중 일부를 발췌한 코드임. 실제 파일에서는 기존 설정의 필요한 값만 변경함.

```python
TOTAL_TIMESTEPS = 200_000

# HPARAMS 중 주요 항목 발췌
HPARAMS = dict(
    lr=1e-3,
    n_steps=2048,
    n_epochs=10,
    minibatch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    hidden_sizes=(256, 256),
    activation="relu",
    policy_type="hybrid",
    seed=0,
    device="auto",
)
```

| 설정 | 코드에서 활용되는 방식 |
|---|---|
| `TOTAL_TIMESTEPS` | `learn()`에 전달하는 총 학습 환경 스텝 목표 |
| `lr` | PPO 내부 Adam optimizer의 학습률 |
| `n_steps` | 업데이트 전 수집하는 환경 스텝 수 |
| `n_epochs` | 수집한 데이터에 대한 업데이트 반복 횟수 |
| `minibatch_size` | 한 번의 optimizer 업데이트에 사용하는 sample 수 |
| `hidden_sizes`, `activation` | 신경망 은닉층 크기 및 활성화 함수 |
| `policy_type` | 기본 `hybrid`: 가감속 연속 출력과 차선 선택 이산 출력 |
| `seed` | NumPy와 PyTorch 난수 seed |
| `device` | `auto`는 CUDA가 있으면 CUDA, 없으면 CPU 선택 |

현재 PPO의 `auto`는 BC와 달리 MPS를 자동 선택하지 않음.

```python
EVAL_INTERVAL = 5
EVAL_EPISODES = 5
LOG_INTERVAL = 1
RESULTS_DIR = "results"
RUN_NAME = None
```

`EVAL_INTERVAL=5`는 모델 업데이트 5회마다 평가한다는 의미임. 평가 한 번에 `EVAL_EPISODES`만큼 주행함. `LOG_INTERVAL`은 콘솔 출력 주기이며 TensorBoard 기록은 매 업데이트 수행함.

`RUN_NAME=None`이면 폴더 이름을 자동 생성함. 실험 이름을 지정하려면 `RUN_NAME="exp_lr1e-4"`처럼 수정하되 기존 결과와 겹치지 않는 이름을 사용함.

## 4. Reward 설정을 환경에 전달하기

기본 reward는 `env/mdp_config.py`에 있으며, 실험에서 바꿀 값은 `train.py`의 `REWARD_OVERRIDES`에 작성함.

```python
REWARD_OVERRIDES = dict(
    speed_weight=0.1,
    collision_penalty=5.0,
    arrival_bonus=2.0,
    close_gap_threshold=6.0,
    close_gap_penalty=0.1,
    blocked_penalty=0.05,
    blocked_gap=20.0,
    blocked_speed_frac=0.7,
    lane_change_penalty=0.0,
    invalid_action_penalty=0.0,
)
REWARD = {**REWARD_DEFAULTS, **REWARD_OVERRIDES}
```

합친 `REWARD`를 환경 생성 시 전달하고, `env/sumo_env.py`에서 실제 스텝별 보상을 계산함. `test.py`도 현재 `train.py`의 `REWARD`를 가져옴. 보상 계수를 바꾼 실험은 평가 reward뿐 아니라 충돌률·완주율을 함께 비교함.

## 5. 도로·환경·PPO 에이전트 생성

`train.py`는 먼저 도로 설정을 읽어 SUMO 파일을 생성함.

```python
sumocfg = road_builder.build(
    ROAD, EGO, TRAFFIC,
    os.path.join(BASE, "env", "sumo"),
)
```

생성한 설정 파일 경로와 관측·행동·reward 설정으로 환경을 생성함.

```python
env = SumoHighwayEnv(
    cfg_path=sumocfg,
    road=ROAD, ego=EGO,
    mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
    action=ACTION, reward=REWARD,
    traffic=TRAFFIC,
    gui=False,
)
```

이어서 `utils/buffer.py`의 `RolloutBuffer`와 `algorithms/ppo.py`의 `PPO`를 생성함.

```python
from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer

buffer = RolloutBuffer()
agent = PPO(
    obs_dim=env.observation_space.shape[0],
    act_dim=env.action_space.shape[0],
    buffer=buffer,
    **HPARAMS,
)
```

관측·행동 차원은 환경에서 읽고, `HPARAMS`는 PPO 생성자의 인자로 전달함. `RolloutBuffer`는 관측, 행동, 행동의 log probability, reward, 가치 추정값, 종료 여부를 보관함.

## 6. Policy 신경망은 어디에 있나요?

`algorithms/ppo.py`는 `policy_type="hybrid"`일 때 `utils/networks.py`의 `HybridActorCritic`을 생성함. 현재 설정은 관측을 두 개의 256차원 ReLU 은닉층에 통과시키고 세 출력 head로 전달함.

```python
# utils/networks.py: HybridActorCritic 출력 head
self.mu_head = orthogonal_init(nn.Linear(last, 1), gain=0.01)
self.lane_head = orthogonal_init(nn.Linear(last, 3), gain=0.01)
self.v_head = orthogonal_init(nn.Linear(last, 1), gain=1.0)
```

`mu_head`는 가감속 평균, `lane_head`는 세 차선 명령의 점수, `v_head`는 가치 추정값을 출력함. 최종 행동은 `[accel_raw, lane_change_raw]`이며 환경이 실제 차량 제어에 사용함.

학습 행동은 `PPO.act()`로 선택하고 평가에서는 `PPO.predict(obs, deterministic=True)`를 사용함. 기본 hybrid 모델의 평가 행동은 가감속 평균과 가장 점수가 높은 차선 명령으로 결정함.

## 7. PPO 학습 루프 읽기

### 7.1 train.py에서 learn() 호출

```python
agent.learn(
    env,
    total_timesteps=TOTAL_TIMESTEPS,
    logger=logger,
    log_interval=LOG_INTERVAL,
    eval_interval=EVAL_INTERVAL,
    eval_episodes=EVAL_EPISODES,
)
```

실제 학습은 `algorithms/ppo.py`의 `PPO.learn()`이 담당함. 다음은 핵심 순서의 발췌임.

```python
obs, last_value, finished = self.collect_rollout(
    env, obs, logger=logger, ep_stats=ep_stats,
)
advantages, returns = self.buffer.compute_gae(
    last_value, self.gamma, self.gae_lambda,
)
stats = self.update(advantages, returns)
```

`collect_rollout()`은 `n_steps`만큼 주행하고, `utils/buffer.py`의 `compute_gae()`는 정책·가치 학습에 사용할 값을 계산하며, `update()`는 신경망을 수정함. 총 학습 스텝 목표에 도달할 때까지 반복함. `n_steps` 단위로 수집하므로 최종 스텝 수는 목표를 조금 넘을 수 있음.

### 7.2 collect_rollout(): SUMO와 상호작용

```python
action, log_prob, value = self.act(obs)
next_obs, reward, terminated, truncated, _ = env.step(action)

self.num_timesteps += 1
ep_stats[0] += reward
ep_stats[1] += 1
```

`env.step()`은 다음 관측, reward, 종료 여부 두 가지, 부가 정보를 반환함. 충돌·도착 등의 종료는 `terminated`, 시간 제한 종료는 `truncated`로 구분함. 실제 저장 코드는 시간 제한 종료에 대한 보정을 포함함.

```python
stored_reward = reward
if truncated and not terminated:
    stored_reward += self.gamma * self.get_value(next_obs)

self.buffer.add(
    obs, action, log_prob, stored_reward, value, terminated,
)
```

에피소드가 끝나면 reward 합과 길이를 기록하고 `env.reset()`으로 다음 주행을 시작함. 한 rollout에 여러 에피소드가 들어갈 수도 있고 에피소드가 다음 rollout까지 이어질 수도 있음. 다음 rollout 시작 시 `self.buffer.clear()`로 이전 수집 기록을 비움.

### 7.3 update(): loss 계산과 optimizer 실행

`algorithms/ppo.py`의 `update()`는 저장한 데이터를 미니배치로 나누고 현재 모델로 행동을 다시 평가함.

```python
log_prob, entropy, value = self.policy.evaluate_actions(
    obs_t[mb], act_t[mb],
)
ratio = torch.exp(log_prob - old_log_prob_t[mb])

surrogate_1 = ratio * adv_t[mb]
surrogate_2 = torch.clamp(
    ratio, 1 - self.clip_eps, 1 + self.clip_eps,
) * adv_t[mb]

policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
value_loss = ((value - ret_t[mb]) ** 2).mean()
entropy_loss = -entropy.mean()

loss = (
    policy_loss
    + self.value_coef * value_loss
    + self.entropy_coef * entropy_loss
)
```

`policy_loss`는 행동 선택 모델, `value_loss`는 가치 출력의 학습에 사용함. 현재 hybrid 설정에서는 차선 head 항도 추가됨.

```python
if self.lane_entropy_coef > 0 and hasattr(self.policy, "lane_entropy"):
    loss = loss - (
        self.lane_entropy_coef
        * self.policy.lane_entropy(obs_t[mb]).mean()
    )
```

BC와 마찬가지로 loss를 역전파하고 optimizer를 실행하며, PPO에서는 gradient 크기 제한도 적용함.

```python
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.optimizer.step()
```

미니배치 업데이트를 `n_epochs`번 반복하고 평균 loss 및 진단값을 로거에 전달함.

## 8. 콘솔과 TensorBoard에서 학습 확인

콘솔 출력 예시이며 수치는 실행마다 달라짐.

```text
[update   11] steps=  22528  ep_rew_mean=18.22  std=0.5909  policy_loss=-0.005901  value_loss=0.4878 ...
```

`steps`는 누적 학습 환경 스텝 수, `update`는 모델 업데이트 횟수임. `ep_rew_mean`은 최근 완료된 최대 20개 학습 에피소드의 평균 reward이며, 완료된 에피소드가 없으면 `nan`이 표시될 수 있음.

학습 중 다른 터미널에서 같은 가상환경을 활성화하고 프로젝트 최상위 폴더로 이동한 뒤 실행함. 아래 `cd`는 본인의 실제 경로로 변경함.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sumo-rl
cd /프로젝트의/실제/경로/Artificial-Intelligence-and-Control-for-Autonomous-Driving
tensorboard --logdir results
```

브라우저에서 <http://localhost:6006>에 접속하고 Scalars를 확인함. x축은 BC의 epoch와 달리 누적 학습 환경 스텝 수임.

| TensorBoard 태그 | 내용 |
|---|---|
| `train/policy_loss` | 정책 loss |
| `train/value_loss` | 가치 출력 loss |
| `train/ep_rew_mean` | 최근 최대 20개 학습 에피소드의 평균 reward |
| `episode/return`, `episode/length` | 개별 학습 에피소드의 reward 합과 스텝 수 |
| `eval/ep_return` | 평가 에피소드 평균 reward |
| `eval/collision_rate`, `eval/success_rate` | 평가 충돌률·완주율 |
| `eval/mean_speed`, `eval/lane_changes` | 평가 평균 속도·에피소드당 실제 차선 변경 횟수 |
| `train/entropy`, `train/approx_kl`, `train/clip_frac` | PPO 업데이트 진단값 |
| `train/p_lane_change` | hybrid 모델의 평균 차선 변경 선택 확률 |

PPO loss는 BC의 MSE와 목적이 다르므로 loss 감소만으로 주행 성능을 판단하지 않음. 평가 reward와 충돌률·완주율을 함께 확인함.

`No dashboards are active`가 표시되면 로그 경로를 확인함. 상대경로 `results`는 TensorBoard를 실행한 터미널의 현재 폴더를 기준으로 해석됨. 실제 절대경로를 지정해도 됨.

```bash
tensorboard --logdir="/프로젝트의/실제/절대경로/results" --reload_interval=2
```

`Retrying in 1 seconds`는 에피소드 초기화 시 새 SUMO에 TraCI가 연결을 재시도하는 메시지임. 이후 update와 steps가 증가한다면 학습이 진행 중임.

## 9. 주기적 평가와 저장 결과

`PPO.learn()`은 설정한 업데이트 주기마다 `utils/evaluator.py`의 함수를 호출함.

```python
m = evaluate_policy(self, env, n_episodes=eval_episodes)
logger.log_eval(
    self.num_timesteps, m.as_dict(), m.summary(), eval_episodes,
)
```

평가 함수는 `predict(obs, deterministic=True)`로 행동을 결정하고 여러 에피소드의 충돌률·완주율·평균 속도·reward 등을 기록함. 같은 환경 객체를 사용하므로 평가 후 학습 환경을 reset함.

학습 종료 또는 Ctrl+C 중단 시 `train.py`의 `finally` 블록에서 환경·로거를 닫고 모델·설정을 저장함.

```python
env.close()
logger.close()
model_path = os.path.join(run_dir, "model.pt")
agent.save(model_path)
snapshot_configs(run_dir)
```

```text
results/run_YYYYMMDD_HHMMSS/
├── model.pt
├── training_log.csv
├── eval_log.csv
├── train.py
├── road_config.py
├── mdp_config.py
└── tb/
    └── events.out.tfevents.*
```

| 파일 | 내용 |
|---|---|
| `model.pt` | 종료 시점 정책 신경망의 `state_dict` |
| `training_log.csv` | 업데이트별 스텝 수, 평균 reward, loss·진단값 |
| `eval_log.csv` | 평가 결과. 평가가 수행된 경우 생성 |
| 복사된 Python 설정 파일 | 해당 실험의 학습·도로·MDP 설정 |
| `tb/events.out.tfevents.*` | 학습 중 기록하는 TensorBoard 이벤트 |

CSV는 `logger.close()` 또는 `save_csv()`에서 저장함. PPO의 `model.pt`는 BC처럼 최저 validation loss 모델을 선택한 것이 아니라 종료 시점 모델임. optimizer 상태를 포함한 학습 재개용 체크포인트는 아님.

## 10. 저장한 모델로 주행 평가하기

학습 종료 시 출력되는 실제 모델 경로를 사용함. 아래 폴더 이름은 본인의 결과 폴더로 변경함.

```bash
python test.py results/run_YYYYMMDD_HHMMSS/model.pt --episodes 5
```

화면 없이 수치만 평가하려면 다음과 같이 실행함.

```bash
python test.py results/run_YYYYMMDD_HHMMSS/model.pt --algorithm ppo --episodes 5 --nogui
```

`test.py`는 기본적으로 모델 유형을 자동 판별함. 과거 모델 평가 시 현재 네트워크 설정과 학습 당시 구조가 맞는지 확인함. 도로나 reward를 변경한 경우에도 저장된 설정 파일과 현재 파일을 비교함.

실습에서는 학습률이나 reward 계수를 바꾼 모델을 서로 다른 결과 폴더에 저장하고 TensorBoard 평가 지표와 실제 SUMO 주행을 함께 비교함. 충돌·완주 여부, 앞차 접근 거리, 불필요한 가감속, 실제 차선 변경을 확인함.
