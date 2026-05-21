#!/bin/bash
set -euo pipefail

REAL_BASIC_VSR_DIR="${REAL_BASIC_VSR_REPO:-/content/RealBasicVSR}"
CHECKPOINT_PATH="${REAL_BASIC_VSR_CHECKPOINT:-${REAL_BASIC_VSR_DIR}/checkpoints/RealBasicVSR_x4.pth}"
MMCV_URL="https://files.pythonhosted.org/packages/88/ef/862b968b07ae49c21e4d71984e5e9da887d24ab117e40980405b8e228d1d/mmcv-1.7.2.tar.gz"
CHECKPOINT_FILE_ID="1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID"

echo "[Crop] Setup RealBasicVSR for Colab."

python -m pip install -q ffmpeg-python==0.2.0 imageai==3.0.3 gdown addict yapf lmdb av
python -m pip install -q mmedit==0.16.1 --no-deps

if [ ! -d "${REAL_BASIC_VSR_DIR}/.git" ]; then
    rm -rf "${REAL_BASIC_VSR_DIR}"
    git clone https://github.com/ckkelvinchan/RealBasicVSR.git "${REAL_BASIC_VSR_DIR}"
fi

MMCV_TMP_DIR="$(mktemp -d)"
curl -L -o "${MMCV_TMP_DIR}/mmcv-1.7.2.tar.gz" "${MMCV_URL}"
tar -xzf "${MMCV_TMP_DIR}/mmcv-1.7.2.tar.gz" -C "${MMCV_TMP_DIR}"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE_PACKAGES}/mmcv"
cp -r "${MMCV_TMP_DIR}/mmcv-1.7.2/mmcv" "${SITE_PACKAGES}/mmcv"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"
if [ ! -s "${CHECKPOINT_PATH}" ]; then
    gdown --id "${CHECKPOINT_FILE_ID}" -O "${CHECKPOINT_PATH}"
fi

python - <<'PY'
import importlib.util
import os

import mmcv
import mmedit

required_modules = ["ffmpeg", "imageai", "cv2", "torch", "mmcv", "mmedit"]
for module_name in required_modules:
    print(f"{module_name}: {importlib.util.find_spec(module_name) is not None}")

print("mmcv:", mmcv.__version__)
print("mmedit:", mmedit.__version__)
print("RealBasicVSR script:", os.path.exists("/content/RealBasicVSR/inference_realbasicvsr.py"))
print("RealBasicVSR config:", os.path.exists("/content/RealBasicVSR/configs/realbasicvsr_x4.py"))
print("RealBasicVSR checkpoint:", os.path.exists("/content/RealBasicVSR/checkpoints/RealBasicVSR_x4.pth"))
PY
