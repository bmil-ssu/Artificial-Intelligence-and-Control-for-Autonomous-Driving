# SUMO 차량 RL 트레이닝 (수업용 · PPO 직접 구현 · 연속 행동)

3차선 고속도로에서 ego 차량(빨간 차)이 주변 교통류 속에서 연속 가속도 제어로
"빠르되 충돌 없이" 달리도록 PPO로 학습하는 프로젝트.
**Windows / macOS 공용** — 코드는 동일하고 SUMO 설치 방법만 다릅니다.

## 구조

```
sumo_rl/
├─ train.py                 ← 학습 진입점 (관리자 역할만)
├─ test.py                  ← 학습 결과를 sumo-gui로 재생
├─ view_road.py             ← 도로 구조만 GUI로 확인 (학습 전 도로 검증용)
├─ setup_mac.sh             ← macOS 자동 설치 스크립트
│
├─ env/                     ← 환경(문제 정의) 관련 전부
│  ├─ road_config.py        도로/차량/교통류 설정 (파이썬 dict)
│  ├─ mdp_config.py         MDP 정의: 관측·행동·보상·에피소드 길이
│  ├─ road_builder.py       road_config → SUMO 파일 자동 생성 (netconvert)
│  ├─ sumo_env.py           Gymnasium 환경 (TraCI로 SUMO 조종)
│  └─ sumo/                 자동 생성되는 SUMO 파일 (직접 수정하지 말 것)
│
├─ algorithms/              ← 강화학습 알고리즘 (알고리즘 로직만)
│  └─ ppo.py                PPO 본체: RolloutBuffer + GAE + clipped update
│
├─ utils/                   ← 알고리즘/환경에 독립적인 재사용 부품
│  ├─ networks.py           build_mlp, GaussianActorCritic, CategoricalActorCritic
│  ├─ buffer.py             RolloutBuffer(on-policy) / ReplayBuffer(off-policy)
│  ├─ evaluator.py          evaluate_policy → 주행지표(DrivingMetrics)
│  └─ logger.py             RunLogger: 콘솔 + TensorBoard + CSV
│
└─ results/                 ← 학습 결과 (실행할 때마다 run 폴더가 하나씩 생김)
   └─ run_20260824_153012/
      ├─ model.pt           학습된 가중치 → test.py에 이 경로를 넘겨 재생
      ├─ training_log.csv   업데이트별 학습 곡선 (엑셀/pandas로 열기)
      └─ *_config.py        그 run에 사용된 설정 스냅샷 (재현/비교용)
```

설계 의도: **환경(문제)과 알고리즘(풀이)을 폴더 수준에서 분리**했다.
- 문제를 바꾸는 실험 → env/ 만 수정
- 풀이를 바꾸는 실험 → algorithms/ 만 수정 (새 알고리즘은 ppo.py와
  같은 인터페이스로 파일을 추가하고 train.py의 import 두 줄만 교체)

## 관측 정의 (19차원, [-1,1])

"내 차선 + 양옆 1차선씩" 총 3개 차선의 국소 교통 상황을 본다:

```
[0]        내 절대속도 / vmax
[1..12]    차선별 블록 × 3 (왼쪽 → 현재 → 오른쪽):
             선행차 상대거리/W, 선행차 상대속도/vmax,
             후행차 상대거리/W(음수), 후행차 상대속도/vmax
             (없으면 선행 = +1/+1, 후행 = -1/-1)
[13..15]   차선별 연결성: -1 = 차선 없음 / 1 = 이어짐 /
             0~1 = 그 비율(거리/W)만큼 가면 차선이 끝남 (합류·차선감소 예고)
[16..18]   차선별 전방 차량 밀도: 0(텅 빔) ~ 1(꽉 참). 없는 차선도 1(꽉 참 취급)
```

도로 구조에 대한 하드코딩 없이 매 스텝 SUMO 네트워크를 조회하므로
(getNeighbors, lane.getLinks 등), **도로를 어떻게 바꿔도 관측 코드는 그대로
동작한다.** 자세한 레이아웃은 env/mdp_config.py 주석 참고.

## 행동 정의 (연속)

행동은 **a ∈ [-1, 1] 실수 하나** (가속 페달과 브레이크를 하나의 축으로):

```
a = +1.0  →  최대 가속  +EGO["accel"] m/s²   (기본 5.4)
a =  0.0  →  속도 유지
a = -1.0  →  최대 감속  -EGO["decel"] m/s²   (기본 5.4)
중간값은 비례.  한 스텝 속도 변화량 = 가속도 × step_length
```

연속 행동이므로 PPO는 **Gaussian 정책** π(a|s) = N(μ(s), σ)을 사용한다.
μ는 신경망 출력(tanh로 [-1,1] 제한), σ는 학습되는 파라미터로
초반엔 넓게 탐험하다가 학습이 진행되며 좁아진다. (algorithms/ppo.py 상단 주석 참고)

## 보상 정의

```
매 스텝:  + speed_weight(0.1) × (내 속도 / vmax)          ← 빠를수록 보상
          - close_gap_penalty(0.2)  (앞차 10m 미만 접근 시) ← 위험운전 감점
종료 시:  충돌 -5 / 완주 +2 / 시간초과(400스텝) 추가보상 없음
```

이상적 에피소드 리턴 ≈ +10, 초반 충돌 시 마이너스권.
값과 설계 이유(스케일을 왜 줄였는지)는 env/mdp_config.py의 REWARD 주석 참고.

---

## 설치 (아무것도 설치 안 된 컴퓨터 기준)

### Windows

**1단계. 파이썬 설치** (이미 있으면 생략 — cmd에서 `python --version` 으로 확인)

- https://www.python.org/downloads/ 에서 최신 Python 3.x 인스톨러 다운로드
- 실행 후 첫 화면 맨 아래의 **"Add python.exe to PATH" 체크박스를 반드시 체크**
  (이걸 안 하면 cmd에서 python/pip 명령이 안 먹혀서 제일 고생함)
- "Install Now" 클릭
- 확인: **새** 명령 프롬프트를 열고
  ```
  python --version
  pip --version
  ```
  둘 다 버전이 출력되면 성공

※ 아나콘다를 쓰고 싶다면 대신 https://www.anaconda.com/download 설치 후
   "Anaconda Prompt"에서 아래 과정을 진행해도 된다 (둘 중 하나만 하면 됨).

**2단계. SUMO 설치**

- https://eclipse.dev/sumo/ → Downloads → **Windows installer (64bit)** 다운로드
- 설치 중 **"Set SUMO_HOME"** 옵션 반드시 체크
- 확인: **새** 명령 프롬프트에서
  ```
  echo %SUMO_HOME%
  sumo --version
  ```

환경변수가 안 잡혔다면 (설치 폴더 존재 확인 후):
```
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
setx PATH "%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin"
```
※ `setx`는 **새로 여는** 터미널부터 적용됨. 터미널/IDE 재시작 필수.

**3단계. 파이썬 패키지 설치**

```
python -m pip install torch gymnasium traci sumolib tensorboard
```
(`pip install ...`도 되지만, `python -m pip`가 "지금 이 파이썬에 설치"를
보장해서 파이썬이 여러 개 깔린 컴퓨터에서도 꼬이지 않는다)

### macOS

**자동 (권장)** — 프로젝트 폴더에서 아래 실행. Homebrew만 있으면
파이썬이 없어도 파이썬 → XQuartz → SUMO → 환경변수 → 패키지까지 전부 설치된다:
```bash
chmod +x setup_mac.sh   # 최초 1회
./setup_mac.sh
source ~/.zshrc
```
Homebrew도 없다면 먼저:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**수동:**
```bash
# 0. 파이썬 (이미 있으면 생략 — `python3 --version` 으로 확인)
brew install python

# 1. XQuartz — sumo-gui가 X11 기반이라 필요 (학습만 할 거면 생략 가능)
#    설치 후 로그아웃 → 재로그인 (또는 재부팅) 해야 적용됨
brew install --cask xquartz

# 2. SUMO (공식 Homebrew tap)
brew tap dlr-ts/sumo
brew install sumo

# 3. SUMO_HOME 환경변수
echo "export SUMO_HOME=\"$(brew --prefix sumo)/share/sumo\"" >> ~/.zshrc
source ~/.zshrc

# 4. 파이썬 패키지
python3 -m pip install torch gymnasium traci sumolib tensorboard
```

참고: `brew --prefix sumo`는 Apple Silicon이면 `/opt/homebrew/opt/sumo`,
Intel 맥이면 `/usr/local/opt/sumo`를 반환한다 (위 명령이 알아서 처리).

**공통**: 이 프로젝트의 `env/sumo_env.py`는 SUMO_HOME이 설정 안 돼 있어도
Windows/macOS/Linux의 기본 설치 경로를 자동 탐지하므로,
환경변수를 깜빡해도 대부분 그냥 동작한다.

---

## 실행 (Windows / macOS 동일)

```bash
python view_road.py                              # 도로만 GUI로 띄움 (▶ 눌러야 차 이동)
python view_road.py --netedit                    # netedit(도로 편집기)로 도로망 열기
python train.py                                  # 학습 → results/run_.../ 에 저장
python test.py results/run_.../model.pt          # 특정 run의 모델을 GUI로 재생
python test.py                                   # 경로 생략 시 가장 최근 run 자동 선택
python test.py --episodes 5                      # 재생 에피소드 수 조절
python test.py --nogui --episodes 20             # GUI 없이 주행 지표만 측정 (빠름)
```

`test.py`는 GUI 재생 시 창이 뜬 뒤 **▶(플레이) 버튼을 눌러야 에피소드가
시작**되고, 에피소드마다 창이 새로 뜨면 다시 ▶를 누르면 된다.

`--nogui` 모드는 화면 없이 조용히 주행시키고 지표 요약만 출력한다:

```
─── 주행 지표 (20 에피소드 평균) ───
  충돌률       : 5%
  완주율       : 90%
  시간초과율   : 5%
  평균 속도    : 27.3 m/s (98.3 km/h)
  평균 차간거리: 34.2 m
  최소 차간거리: 8.1 m
  평균 보상    : 9.4
  평균 길이    : 121 스텝
```

GUI 재생은 에피소드가 실시간이라 느리므로, **정량 평가는 --nogui로
에피소드 수를 넉넉히(20~50) 잡는 것을 권장**한다. 학습 중 eval과
완전히 같은 코드(utils/evaluator.py)를 쓰므로 수치가 그대로 비교된다.
(macOS에서 python/pip가 파이썬2를 가리키면 `python3` / `pip3` 사용)

여러 설정으로 실험할 때는 run마다 결과가 분리 저장되고 설정 스냅샷이 함께
남기 때문에, "이 곡선이 어떤 설정에서 나온 건지"를 나중에도 추적할 수 있다.
run 이름을 직접 붙이려면 train.py 상단의 `RUN_NAME`을 지정
(예: "exp_low_traffic").

## 학습 로그 보는 법

**1) 콘솔 (실시간)** — train.py 실행 중 업데이트마다 한 줄씩 출력.
`ep_rew_mean`(최근 20에피소드 평균)이 꾸준히 오르면 학습이 되는 것.

**2) TensorBoard (그래프, 실시간)** — 별도 터미널에서:
```
tensorboard --logdir results
```
→ 브라우저에서 http://localhost:6006 접속.
- `train/ep_rew_mean` : 학습 곡선 (제일 중요)
- `train/std` : 탐험 폭. 서서히 줄면 정상, 초반부터 급락하면 entropy_coef↑
- `episode/return` : 스무딩 전 개별 에피소드 보상 (요동 커도 정상)
- 여러 run이 색깔별로 겹쳐 그려지므로 **설정 바꿔가며 비교 실험**할 때 특히 유용.
  왼쪽 체크박스에서 비교할 run만 선택하면 된다.
- 학습이 돌아가는 중에도 새로고침하면 실시간으로 갱신된다.

**진단 지표 읽는 법** (콘솔/TensorBoard에 함께 출력)
- `std` : 탐험 폭 σ. **꾸준히 줄어들어야 정상.** 계속 1.0 근처면 정책이
  랜덤에 머물고 있다는 뜻 → entropy_coef를 0으로, init_log_std를 낮출 것
- `approx_kl` : 업데이트당 정책 변화량. 0.02를 크게 넘으면 lr을 낮출 것
- `clip_frac` : 클리핑에 걸린 비율. 0.1~0.3이 정상 범위
- `value_loss` : critic 오차. 계속 커지면 보상 스케일이 너무 큰 것

**3) 주기적 평가 — 주행 지표** (eval_interval 업데이트마다 자동 실행)

학습 중 보상은 탐험 노이즈가 섞여 있어 실제 실력과 다를 수 있으므로,
주기적으로 **결정적 정책(노이즈 없이 μ만)**으로 몇 에피소드를 주행시켜
주행 지표를 잰다 (설정: train.py 상단의 EVAL_INTERVAL / EVAL_EPISODES):

| 지표 | 의미 |
|---|---|
| eval/collision_rate | 충돌로 끝난 에피소드 비율 (0~1, 낮을수록 좋음) |
| eval/success_rate | 도로 끝까지 완주한 비율 (1-충돌률-완주율 = 시간초과 비율) |
| eval/mean_speed | 전 스텝 평균 주행속도 (m/s) |
| eval/mean_gap | 선행차가 감지된 스텝에서의 평균 차간거리 (m) |
| eval/ep_return | 에피소드당 평균 누적 보상 |

콘솔에도 `[eval @스텝] 충돌률=.. 완주율=.. ...` 형태로 출력되고,
TensorBoard의 eval/* 그래프와 `results/run_.../eval_log.csv` 에 기록된다.
**학습이 잘 되면 collision_rate ↓, success_rate ↑, mean_speed ↑ 로 움직인다.**
mean_gap이 REWARD["close_gap_threshold"](기본 10m) 근처까지 붙는다면
공격적인 정책, 크게 유지된다면 보수적인 정책이라는 해석도 가능.

**4) CSV (학습 후)** — `results/run_.../training_log.csv`(학습 곡선)과
`eval_log.csv`(주행 지표)를 엑셀/pandas로 열기.
같은 내용이 텍스트로 남으므로 보고서용 그래프 그릴 때 편하다.

## 무엇을 어디서 바꾸나

| 바꾸고 싶은 것 | 파일 | 항목 |
|---|---|---|
| 도로 길이/차선 수 | env/road_config.py | ROAD (바꾼 뒤 `python view_road.py`로 확인) |
| 교통 밀도(난이도) | env/road_config.py | TRAFFIC["vehs_per_hour"] |
| 가속/감속 한계 (행동 스케일) | env/road_config.py | EGO["accel"], EGO["decel"] |
| 보상 설계 | env/mdp_config.py | REWARD |
| 에피소드 길이 | env/mdp_config.py | SIMULATION["max_steps"] |
| 학습량/학습률/gamma 등 | **train.py 상단** | TOTAL_TIMESTEPS, HPARAMS |
| 네트워크 구조 | train.py 상단 | HPARAMS의 hidden_sizes, activation |

## 수업에서 읽는 순서 추천

1. `env/mdp_config.py` — 이 문제의 MDP가 무엇인지 (S, A, R)
2. `env/sumo_env.py` — MDP가 코드로 어떻게 구현되는지 (reset/step/보상)
3. `utils/networks.py` — 정책/가치 네트워크가 어떻게 생겼는지
4. `algorithms/ppo.py` — 알고리즘. 맨 위 주석에 PPO 수식 요약,
   learn() 안이 1) rollout → 2) GAE → 3) update → 4) eval 로 구획됨
5. `train.py` — 전체가 어떻게 조립되는지 + 하이퍼파라미터

## 실험 아이디어 (과제용)

- REWARD["close_gap_penalty"]를 0으로 → 정책이 얼마나 공격적으로 변하는가?
- TRAFFIC["vehs_per_hour"]를 600 / 3000으로 → 난이도에 따른 학습 속도 비교
- EGO["decel"]을 2.7로 낮춤 → 제동력이 약할 때 정책이 얼마나 보수적으로 변하는가?
- entropy_coef를 0.01로 올림 → σ가 안 줄어들어 정책이 랜덤에 머무는 현상 관찰
- init_log_std를 0.0으로 → 초기 탐험이 과하면 학습이 어떻게 되는가
- gae_lambda를 1.0 / 0.0으로 → 분산-편향 트레이드오프 체감

## 자주 나는 문제

공통:
- `connection closed` 오류: 남아있는 sumo 프로세스 종료
  (Windows: 작업관리자에서 sumo.exe / macOS: `pkill sumo`)
- 보상이 요동: train.py의 HPARAMS에서 lr을 낮추거나(1e-4→5e-5) n_steps를 4096으로
- SUMO_HOME 오류: 환경변수 설정 후 터미널/IDE **재시작**

Windows:
- `'sumo'은(는) 내부 또는 외부 명령...`: 설치 안 됐거나 PATH 누락 → 설치 섹션 참고

macOS:
- `sumo-gui` 창이 안 뜨거나 X11 오류: XQuartz 설치 후 **재로그인 안 한 경우**가
  대부분. 학습(train.py)은 GUI가 필요 없으므로 XQuartz 없이도 된다.
- `command not found: sumo`: 새 터미널을 열 것. Apple Silicon에서 brew가
  PATH에 없다면 `~/.zshrc`에 `eval "$(/opt/homebrew/bin/brew shellenv)"` 확인
- pip 권한 오류: `python3 -m venv venv && source venv/bin/activate` 후 설치
- Apple Silicon torch: 일반 `pip3 install torch`면 됨. 이 프로젝트는 CPU로도
  충분히 빠름 (ppo.py는 cuda만 자동 감지, 맥에선 CPU 사용)
