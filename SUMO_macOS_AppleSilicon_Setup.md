# macOS Apple Silicon: SUMO 1.27.1 + RL 설치 가이드

Apple Silicon Mac에서 `SUMO 1.27.1`, `sumo-gui`, Python/TraCI 기반 RL
환경을 구성하는 방법입니다.

## 1. ARM64 확인

``` bash
uname -m
arch
echo $SHELL
sysctl -in sysctl.proc_translated
```

정상 예시:

``` text
arm64
arm64
/bin/zsh
0
```

`x86_64`라면 ARM64 zsh로 변경합니다.

``` bash
arch -arm64 /bin/zsh
chsh -s /bin/zsh
```

Terminal을 완전히 종료 후 다시 실행합니다.

## 2. Xcode Command Line Tools

``` bash
xcode-select --install
```

설치 확인:

``` bash
xcode-select -p
clang --version
pkgutil --pkg-info=com.apple.pkg.CLTools_Executables
```

## 3. XQuartz 설치

XQuartz를 설치한 뒤 Mac에서 로그아웃 → 로그인 또는 재부팅합니다. <https://www.xquartz.org/>

실행 및 DISPLAY 설정:

``` bash
open -a XQuartz
export DISPLAY="$(launchctl getenv DISPLAY)"
```

확인:

``` bash
glxinfo | head -30
glxgears
```

## 4. MacPorts 및 빌드 패키지 설치

MacPorts를 설치한 뒤: <https://www.macports.org/install.php>

본인 맥북의 환경설정 > 정보 > macOS에서 이름이 Tahoe일 경우 macOS Tahoe v26 설치

아래 실행:

``` bash
sudo port selfupdate
sudo port install mesa +llvm fox xercesc3 proj gdal gl2ps cmake ninja
```

확인:

``` bash
which cmake
which ninja
which fox-config
```

모두 `/opt/local/...` 경로를 사용하는지 확인합니다.

## 5. SUMO 1.27.1 소스 빌드

``` bash
conda deactivate

export PATH=/opt/local/bin:/opt/local/sbin:/usr/bin:/bin:/usr/sbin:/sbin

mkdir -p ~/src
cd ~/src
git clone --recursive --branch v1_27_1 https://github.com/eclipse-sumo/sumo.git sumo-1.27.1-mesa
cd sumo-1.27.1-mesa
```

CMake 설정:

``` bash
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

OpenGL 링크 확인:

``` bash
otool -L bin/sumo-gui | grep -iE 'GL|FOX'
```

`libFOX`, `libGL`, `libGLU`가 `/opt/local/lib/...`을 가리켜야 합니다.

## 6. SUMO 환경변수 설정

``` bash
echo 'export SUMO_HOME="$HOME/src/sumo-1.27.1-mesa"' >> ~/.zshrc
echo 'export PATH="$SUMO_HOME/bin:$PATH"' >> ~/.zshrc
echo 'export DISPLAY="$(launchctl getenv DISPLAY)"' >> ~/.zshrc
source ~/.zshrc
```

확인:

``` bash
which sumo
which sumo-gui
sumo --version
```

## 7. ARM64 Miniforge + Python 환경

Apple Silicon용 `Miniforge3-MacOSX-arm64`를 설치합니다. <https://github.com/conda-forge/miniforge/releases/tag/26.5.3-0>

링크에서 Miniforge3-MacOSX-arm64.pkg 설치

필요하면:

``` bash
source ~/miniforge3/etc/profile.d/conda.sh
```

확인:

``` bash
conda info | grep platform
```

반드시 `osx-arm64`가 나와야 합니다.

환경 생성:

``` bash
conda create -n sumo-rl python=3.10 -y
conda activate sumo-rl
```

확인:

``` bash
python -c "import platform; print(platform.machine())"
```

`arm64`가 나와야 합니다.

## 8. Python 패키지 설치

``` bash
python -m pip install --upgrade pip
python -m pip install torch gymnasium traci sumolib tensorboard
```

확인:

``` bash
python -c "import torch, gymnasium, traci, sumolib; print('All imports OK')"
```

## 9. 실행

``` bash
conda activate sumo-rl
cd ~/Artificial-Intelligence-and-Control-for-Autonomous-Driving
python view_road.py
```

이후에는 위 세 명령만 실행하면 됩니다.

## 핵심

macOS에서 `GLXBadContext`가 발생할 경우, SUMO GUI가
`/opt/X11/lib/libGL`이 아니라 MacPorts의 `/opt/local/lib/libGL`을
사용하도록 SUMO를 다시 빌드하는 것이 핵심입니다.
