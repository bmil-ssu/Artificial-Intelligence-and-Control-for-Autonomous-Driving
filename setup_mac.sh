#!/bin/bash
# ─────────────────────────────────────────────────────────────
# macOS용 SUMO + 파이썬 환경 자동 설정 스크립트
#
# 사용법:
#   chmod +x setup_mac.sh    ← 최초 1회, 실행 권한 부여
#   ./setup_mac.sh
#
# 하는 일:
#   1. Homebrew 확인 (없으면 설치 안내 후 종료)
#   2. 파이썬 설치       (없는 경우에만)
#   3. XQuartz 설치      (sumo-gui가 X11 기반이라 필요)
#   4. SUMO 설치         (brew tap dlr-ts/sumo)
#   5. SUMO_HOME 환경변수를 셸 설정 파일에 추가
#   6. 파이썬 패키지 설치
# ─────────────────────────────────────────────────────────────
set -e  # 중간에 하나라도 실패하면 즉시 중단

# 1) Homebrew 확인
if ! command -v brew &> /dev/null; then
    echo "[오류] Homebrew가 없습니다. 먼저 설치하세요:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
fi
echo "[1/6] Homebrew 확인 완료"

# 2) 파이썬 (없으면 brew로 설치)
if ! command -v python3 &> /dev/null; then
    echo "[2/6] 파이썬 설치 중..."
    brew install python
else
    echo "[2/6] 파이썬 이미 설치됨: $(python3 --version)"
fi

# 3) XQuartz (sumo-gui / netedit 실행에 필요한 X11 서버)
if ! brew list --cask xquartz &> /dev/null; then
    echo "[3/6] XQuartz 설치 중..."
    brew install --cask xquartz
    echo "  ※ XQuartz는 설치 후 '로그아웃 → 재로그인'(또는 재부팅)해야 적용됩니다."
else
    echo "[3/6] XQuartz 이미 설치됨"
fi

# 4) SUMO 설치 (공식 Homebrew tap)
if ! command -v sumo &> /dev/null; then
    echo "[4/6] SUMO 설치 중..."
    brew tap dlr-ts/sumo
    brew install sumo
else
    echo "[4/6] SUMO 이미 설치됨: $(sumo --version 2>/dev/null | head -1)"
fi

# 5) SUMO_HOME 환경변수 등록
#    brew --prefix sumo → Apple Silicon: /opt/homebrew/opt/sumo
#                         Intel        : /usr/local/opt/sumo
SUMO_HOME_PATH="$(brew --prefix sumo)/share/sumo"
# 기본 셸이 zsh(맥 기본)면 ~/.zshrc, bash면 ~/.bash_profile
SHELL_RC="$HOME/.zshrc"
[[ "$SHELL" == */bash ]] && SHELL_RC="$HOME/.bash_profile"

if ! grep -q "SUMO_HOME" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# SUMO (setup_mac.sh 가 추가함)" >> "$SHELL_RC"
    echo "export SUMO_HOME=\"$SUMO_HOME_PATH\"" >> "$SHELL_RC"
    echo "[5/6] SUMO_HOME을 $SHELL_RC 에 추가: $SUMO_HOME_PATH"
else
    echo "[5/6] SUMO_HOME 이미 $SHELL_RC 에 있음"
fi
export SUMO_HOME="$SUMO_HOME_PATH"   # 현재 셸에도 즉시 적용

# 6) 파이썬 패키지
echo "[6/6] 파이썬 패키지 설치 중..."
python3 -m pip install torch gymnasium traci sumolib tensorboard

echo ""
echo "════════════════════════════════════════════"
echo " 설정 완료!  새 터미널을 열거나 아래를 실행:"
echo "   source $SHELL_RC"
echo " 그 다음:"
echo "   python3 train.py    # 학습"
echo "   python3 test.py     # GUI 재생 (XQuartz 재로그인 필요할 수 있음)"
echo "════════════════════════════════════════════"
