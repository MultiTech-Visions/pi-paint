#!/bin/bash
# pi-paint VJ — pull the latest version from GitHub.
# Preserves your assets/ folder and venv/ so you keep your video files
# and don't have to reinstall Python deps.
# Double-click and choose "Execute in Terminal".

set -e
cd "$(dirname "$0")"

BRANCH="claude/charming-volta-rHxFb"
REPO="MultiTech-Visions/pi-paint"

echo ""
echo "============================================================"
echo "  pi-paint VJ — Update from GitHub ($BRANCH)"
echo "============================================================"
echo ""

if [ -d "../.git" ]; then
  echo "[git mode] Detected git checkout. Pulling latest..."
  cd ..
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
  echo "Done."
else
  echo "[zip mode] Downloading latest code from GitHub..."
  URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"
  TMP=$(mktemp -d)
  if ! curl -fSL "$URL" -o "$TMP/vj.zip"; then
    echo ""
    echo "Download failed. Check your internet connection and try again."
    rm -rf "$TMP"
    read -p "Press Enter to close..."
    exit 1
  fi
  unzip -q "$TMP/vj.zip" -d "$TMP"
  SRC=$(ls -d "$TMP"/pi-paint-*/vj 2>/dev/null | head -1)
  if [ -z "$SRC" ]; then
    echo "Couldn't find vj/ inside the downloaded ZIP. Aborting."
    rm -rf "$TMP"
    read -p "Press Enter to close..."
    exit 1
  fi
  echo "Replacing code files (your assets/ and venv/ are preserved)..."
  for item in "$SRC"/*; do
    name=$(basename "$item")
    case "$name" in
      assets|venv) continue ;;
    esac
    cp -rf "$item" ./
  done
  chmod +x ./*.sh 2>/dev/null || true
  rm -rf "$TMP"
  echo "Done."
fi

echo ""
echo "If requirements.txt changed, re-run setup.sh to pick up new deps."
echo ""
read -p "Press Enter to close..."
