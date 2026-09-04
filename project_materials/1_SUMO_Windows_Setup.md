# Windows: SUMO + RL 설치 가이드

Windows에서 `SUMO`, `sumo-gui`, Python/TraCI 기반 RL 환경을 구성하는 방법입니다.

## 0. 프로젝트 다운로드

명령 프롬프트(cmd) 또는 PowerShell에서 아래 명령어를 실행합니다.

```bash
git clone https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving.git
```

---

## 1. Python 설치

먼저 Python 설치 여부를 확인합니다.

```bat
python --version
```

버전이 정상적으로 출력되면 이미 설치되어 있으므로 이 단계를 생략합니다.
Python이 설치되어 있지 않다면 다음 순서로 진행합니다.

1. [Python 공식 홈페이지](https://www.python.org/downloads/)에서 최신 Python 3.x 인스톨러를 다운로드합니다.
2. 인스톨러 실행 후 첫 화면 맨 아래의 **`Add python.exe to PATH`** 체크박스를 반드시 체크합니다.
   - 이 옵션을 체크하지 않으면 cmd에서 `python`, `pip` 명령어가 인식되지 않을 수 있습니다.
3. **Install Now**를 클릭하여 설치합니다.
4. 설치가 끝나면 기존 터미널을 닫고 **새 명령 프롬프트**를 실행합니다.

설치 확인:

```bat
python --version
pip --version
```

두 명령어 모두 버전이 정상적으로 출력되면 설치가 완료된 것입니다.

> **참고:** Anaconda를 사용하고 싶다면 Python 대신
> [Anaconda](https://www.anaconda.com/download)를 설치한 뒤
> **Anaconda Prompt**에서 이후 과정을 진행해도 됩니다.
> Python 또는 Anaconda 중 하나만 사용하면 됩니다.

---

## 2. SUMO 설치

SUMO 공식 인스톨러를 사용하여 설치합니다.

1. [SUMO 공식 홈페이지](https://eclipse.dev/sumo/)에 접속합니다.
2. **Downloads → Windows installer (64bit)** 를 다운로드합니다.
3. 설치 과정에서 **`Set SUMO_HOME`** 옵션을 반드시 체크합니다.
4. 설치 완료 후 기존 터미널을 닫고 **새 명령 프롬프트**를 실행합니다.

설치 확인:

```bat
echo %SUMO_HOME%
sumo --version
```

`SUMO_HOME` 경로와 SUMO 버전이 출력되면 정상적으로 설치된 것입니다.

### SUMO 환경변수가 설정되지 않은 경우

SUMO가 설치되어 있는데 `%SUMO_HOME%`이 출력되지 않거나 `sumo` 명령어가 인식되지 않는다면,
실제 SUMO 설치 폴더를 확인한 뒤 다음과 같이 환경변수를 설정합니다.

```bat
setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
setx PATH "%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin"
```

> **참고:** `setx`로 설정한 환경변수는 **새로 실행한 터미널부터 적용**됩니다.
> 명령 프롬프트, PowerShell, VS Code 등의 터미널을 모두 닫았다가 다시 실행하세요.

---

## 3. Python 패키지 설치

다음 명령어로 필요한 Python 패키지를 설치합니다.

```bash
python -m pip install torch gymnasium traci sumolib tensorboard
```

> **참고:** 일반적인 `pip install ...` 명령어도 사용할 수 있지만,
> `python -m pip` 형태를 사용하면 **현재 실행 중인 Python 환경에 패키지가 설치되는 것을 명확하게 보장**할 수 있습니다.
> Python이 여러 버전 설치되어 있는 컴퓨터에서 패키지 설치 경로가 꼬이는 문제를 줄일 수 있으므로 이 방식을 권장합니다.

---

## 4. 설치 확인 및 실행

먼저 아래 명령어들이 모두 정상적으로 실행되는지 확인합니다.

```bat
python --version
pip --version
echo %SUMO_HOME%
sumo --version
```

그 다음 프로젝트 폴더로 이동한 뒤 도로 GUI가 정상적으로 실행되는지 확인합니다.

```bash
cd Artificial-Intelligence-and-Control-for-Autonomous-Driving/
python view_road.py
```

`sumo-gui` 창이 정상적으로 열리면 기본 설치가 완료된 것입니다.

---

## 자주 발생하는 문제

### `'sumo'은(는) 내부 또는 외부 명령...` 오류

SUMO가 설치되지 않았거나 SUMO의 `bin` 폴더가 Windows `PATH`에 등록되지 않은 경우입니다.

먼저 다음을 확인합니다.

```bat
echo %SUMO_HOME%
sumo --version
```

필요하면 위의 **SUMO 환경변수가 설정되지 않은 경우** 절차에 따라 `SUMO_HOME`과 `PATH`를 다시 설정합니다.

### `SUMO_HOME`을 설정했는데도 적용되지 않는 경우

환경변수 변경 전부터 열려 있던 터미널에서는 새 설정이 반영되지 않을 수 있습니다.

**명령 프롬프트 / PowerShell / VS Code를 완전히 종료한 뒤 다시 실행**하세요.
