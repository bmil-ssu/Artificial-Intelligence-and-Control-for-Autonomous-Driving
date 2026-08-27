# macOS Apple Silicon: SUMO 1.27.1 + RL 설치 가이드

Apple Silicon Mac에서 `SUMO 1.27.1`, `sumo-gui`, Python/TraCI 기반 RL 환경을 구성하는 방법입니다.

## 1. ARM64 확인

먼저 현재 Terminal이 ARM64 환경에서 실행되고 있는지 확인합니다.

```bash
uname -m
arch
echo $SHELL
sysctl -in sysctl.proc_translated
```

정상 예시:

```text
arm64
arm64
/bin/zsh
0
```

`x86_64`라면 ARM64 zsh로 변경합니다.

```bash
arch -arm64 /bin/zsh
chsh -s /bin/zsh
```

설정 후 Terminal을 완전히 종료한 뒤 다시 실행합니다.

---

## 2. Xcode Command Line Tools 설치

먼저 설치 여부를 확인합니다.

```bash
xcode-select -p
```

`/Library/Developer/CommandLineTools`가 출력되면 이미 설치되어 있으므로 이 단계를 생략합니다.
경로가 없거나 오류가 발생하면 다음 명령어로 설치합니다.

```bash
xcode-select --install
```

설치 확인:

```bash
xcode-select -p
clang --version
pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
```

MacPorts 설치 중 `receipt appears to be missing` 또는
`Neither Xcode nor the Command Line Tools were found` 오류가 발생할 때만
CLT를 재설치합니다.

```bash
sudo rm -rf /Library/Developer/CommandLineTools
sudo xcode-select --reset
xcode-select --install
```

---

## 3. XQuartz 설치

XQuartz 공식 사이트에서 설치합니다.

<https://www.xquartz.org/>

설치 후 **Mac 로그아웃 → 로그인 또는 재부팅**합니다.

XQuartz를 실행합니다.

```bash
open -a XQuartz
```

그 다음 DISPLAY를 설정하고 정상적으로 적용되었는지 확인합니다.

```bash
export DISPLAY="$(launchctl getenv DISPLAY)"
echo $DISPLAY
```

정상 예시:

```text
/var/run/com.apple.launchd.XXXXXXXXXX/org.xquartz:0
```

`org.xquartz:0`이 포함된 경로가 출력되면 정상입니다.

> **참고:** Apple Silicon의 일부 macOS/XQuartz 환경에서는 `glxinfo` 또는
> `glxgears` 실행 시 `BadValue (Apple-DRI)` 오류가 발생할 수 있습니다.
> 본 가이드에서는 이후 MacPorts의 Mesa/OpenGL을 사용하여 SUMO를 직접
> 빌드하므로, 이 단계에서는 `glxinfo`/`glxgears` 테스트를 수행하지 않고
> 다음 단계로 진행합니다.

---

## 4. MacPorts 및 빌드 패키지 설치

[MacPorts 공식 홈페이지](https://www.macports.org/install.php)에서 MacPorts를 설치합니다.

본인 맥북의 **환경설정 → 정보 → macOS**에서 이름이 Tahoe일 경우
macOS Tahoe v26 버전을 설치합니다.

설치 후 다음 명령어를 실행합니다.

```bash
sudo port selfupdate
sudo port install mesa +llvm fox xercesc3 proj gdal gl2ps cmake ninja
```

설치 확인:

```bash
which cmake
which ninja
which fox-config
```

모두 `/opt/local/...` 경로를 사용하는지 확인합니다.

---

## 5. SUMO 1.27.1 소스 빌드

먼저 MacPorts 경로를 우선 사용하도록 설정한 뒤 SUMO 1.27.1 소스를 내려받습니다.

```bash
export PATH=/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin
hash -r

mkdir -p ~/src
cd ~/src
git clone --recursive --branch v1_27_1 https://github.com/eclipse-sumo/sumo.git sumo-1.27.1-mesa
cd sumo-1.27.1-mesa
```

CMake를 설정하고 SUMO를 빌드합니다.

```bash
rm -rf build

cmake -B build -G Ninja \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/local/lib" \
  -DCMAKE_PREFIX_PATH=/opt/local \
  -DCMAKE_IGNORE_PATH="/usr/local;/opt/homebrew" \
  -DCMAKE_FIND_FRAMEWORK=LAST \
  -DOPENGL_gl_LIBRARY=/opt/local/lib/libGL.dylib \
  -DOPENGL_glu_LIBRARY=/opt/local/lib/libGLU.dylib \
  -DOPENGL_INCLUDE_DIR=/opt/local/include \
  -DFOX_CONFIG=/opt/local/bin/fox-config

cmake --build build --parallel $(sysctl -n hw.ncpu)
```

빌드 후 OpenGL 링크를 확인합니다.

```bash
otool -L bin/sumo-gui | grep -iE 'GL|FOX'
```

`libFOX`, `libGL`, `libGLU`가 `/opt/local/lib/...`을 가리켜야 합니다.
`/opt/X11/lib/libGL...`이 나오면 `rm -rf build` 후 CMake 설정부터 다시 빌드합니다.

---

## 6. SUMO 환경변수 설정

다음 명령어로 SUMO 관련 환경변수를 설정합니다.

```bash
echo 'export SUMO_HOME="$HOME/src/sumo-1.27.1-mesa"' >> ~/.zshrc
echo 'export PATH="$SUMO_HOME/bin:$PATH"' >> ~/.zshrc
echo 'export DISPLAY="$(launchctl getenv DISPLAY)"' >> ~/.zshrc
source ~/.zshrc
```

설정 확인:

```bash
which sumo
which sumo-gui
sumo --version
```

---

## 7. ARM64 Miniforge + Python 환경 설치

Apple Silicon용 `Miniforge3-MacOSX-arm64.pkg`를 설치합니다.

<https://github.com/conda-forge/miniforge/releases/latest>

위 링크에서 `Miniforge3-MacOSX-arm64.pkg`를 설치합니다.

필요하면 다음 명령어로 conda를 활성화합니다.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

설치 확인:

```bash
conda info | grep -E 'base environment|platform'
```

정상 예시:

```text
base environment : /Users/<사용자명>/miniforge3
platform : osx-arm64
```

반드시 Miniforge 경로와 `osx-arm64`가 나와야 합니다.
`osx-64`가 나오면 그 상태에서 가상환경을 만들지 마세요.

다음 명령어로 Python 가상환경을 생성하고 활성화합니다.

```bash
conda create -n sumo-rl python=3.10 -y
conda activate sumo-rl
```

환경 확인:

```bash
python -c "import platform; print(platform.machine())"
```

`arm64`가 출력되면 정상입니다.

---

## 8. Python 패키지 설치

다음 명령어로 필요한 Python 패키지를 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install torch gymnasium traci sumolib tensorboard
```

설치 확인:

```bash
python -c "import torch, gymnasium, traci, sumolib; print('All imports OK')"
```

---

## 9. 실행

conda가 활성화되어 있지 않다면 먼저 다음 명령어를 실행합니다.

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

그 다음 가상환경을 활성화하고 프로젝트 폴더로 이동한 뒤 실행합니다.

```bash
conda activate sumo-rl
cd ~/Artificial-Intelligence-and-Control-for-Autonomous-Driving
python view_road.py
```

이후에는 위 세 명령만 실행하면 됩니다.

---

## 핵심

macOS에서 `GLXBadContext`가 발생할 경우, SUMO GUI가
`/opt/X11/lib/libGL`이 아니라 MacPorts의 `/opt/local/lib/libGL`을
사용하도록 SUMO를 다시 빌드하는 것이 핵심입니다.

---

## 자주 발생하는 문제

### MacPorts에서 Xcode 버전 오류가 발생하는 경우

MacPorts 설치 중 다음과 같은 오류가 발생할 수 있습니다.

```text
Error: The installed version of Xcode (15.3) is too old to use on the installed OS version.
Version 26.0 or later is recommended on macOS 26.
Error: Processing of port mesa failed
```

먼저 현재 MacPorts가 어떤 개발자 도구 경로를 사용하고 있는지 확인합니다.

```bash
xcode-select -p
xcodebuild -version
```

예를 들어 다음처럼 나온다면:

```text
/Applications/Xcode.app/Contents/Developer
Xcode 15.3
Build version 15E204a
```

macOS 26에서 오래된 Xcode 15.3이 선택되어 있는 상태입니다.

#### 1. 최신 Command Line Tools가 이미 설치되어 있는 경우

먼저 Command Line Tools가 정상 설치되어 있는지 확인합니다.

```bash
clang --version
pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
```

`clang`의 Target이 `arm64-apple-darwin...`이고 CLT package 정보가 정상 출력된다면,
개발자 경로를 Command Line Tools로 변경합니다.

```bash
sudo xcode-select -s /Library/Developer/CommandLineTools
```

설정 확인:

```bash
xcode-select -p
```

정상 예시:

```text
/Library/Developer/CommandLineTools
```

그 다음 MacPorts 설치를 다시 시도합니다.

```bash
sudo port selfupdate
sudo port install mesa +llvm fox xercesc3 proj gdal gl2ps cmake ninja
```

#### 2. 그래도 동일한 Xcode 버전 오류가 발생하는 경우

macOS 26에서 MacPorts가 여전히 오래된 Xcode를 문제로 판단한다면,
기존 Xcode 15.3 대신 **Xcode 26 이상으로 업데이트**하는 것이 가장 확실합니다.

Xcode 업데이트 후 다음 명령어를 실행합니다.

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

설정 확인:

```bash
xcode-select -p
xcodebuild -version
```

그 다음 MacPorts 설치를 다시 시도합니다.

```bash
sudo port selfupdate
sudo port install mesa +llvm fox xercesc3 proj gdal gl2ps cmake ninja
```

> **핵심:** `xcode-select -p`가 `/Applications/Xcode.app/Contents/Developer`를 가리키면서
> `xcodebuild -version`이 `Xcode 15.3`처럼 오래된 버전을 보여준다면,
> Command Line Tools로 전환하거나 Xcode를 macOS 26에 맞는 최신 버전으로 업데이트해야 합니다.
