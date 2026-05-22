#!/bin/bash
set -euo pipefail

REAL_BASIC_VSR_DIR="${REAL_BASIC_VSR_REPO:-/content/RealBasicVSR}"
CHECKPOINT_PATH="${REAL_BASIC_VSR_CHECKPOINT:-${REAL_BASIC_VSR_DIR}/checkpoints/RealBasicVSR_x4.pth}"
MMCV_URL="https://files.pythonhosted.org/packages/88/ef/862b968b07ae49c21e4d71984e5e9da887d24ab117e40980405b8e228d1d/mmcv-1.7.2.tar.gz"
MMEDIT_URL="https://files.pythonhosted.org/packages/source/m/mmedit/mmedit-0.16.1.tar.gz"
CHECKPOINT_FILE_ID="1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID"

echo "[Crop] Setup RealBasicVSR for Colab."

if [ ! -d "${REAL_BASIC_VSR_DIR}/.git" ]; then
    rm -rf "${REAL_BASIC_VSR_DIR}"
    git clone https://github.com/ckkelvinchan/RealBasicVSR.git "${REAL_BASIC_VSR_DIR}"
fi

SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

MMCV_TMP_DIR="$(mktemp -d)"
curl -L -o "${MMCV_TMP_DIR}/mmcv-1.7.2.tar.gz" "${MMCV_URL}"
tar -xzf "${MMCV_TMP_DIR}/mmcv-1.7.2.tar.gz" -C "${MMCV_TMP_DIR}"
rm -rf "${SITE_PACKAGES}/mmcv"
cp -r "${MMCV_TMP_DIR}/mmcv-1.7.2/mmcv" "${SITE_PACKAGES}/mmcv"

MMEDIT_TMP_DIR="$(mktemp -d)"
curl -L -o "${MMEDIT_TMP_DIR}/mmedit-0.16.1.tar.gz" "${MMEDIT_URL}"
tar -xzf "${MMEDIT_TMP_DIR}/mmedit-0.16.1.tar.gz" -C "${MMEDIT_TMP_DIR}"
rm -rf "${SITE_PACKAGES}/mmedit"
cp -r "${MMEDIT_TMP_DIR}/mmedit-0.16.1/mmedit" "${SITE_PACKAGES}/mmedit"

MMEDIT_SITE_PACKAGES="${SITE_PACKAGES}/mmedit"
cat > "${MMEDIT_SITE_PACKAGES}/core/__init__.py" <<'PY'
# Crop Colab compatibility patch.
# Avoid importing heavyweight training/evaluation utilities. BasicRestorer
# imports psnr, ssim, and InceptionV3 at module import time, but RealBasicVSR
# inference does not use metrics because Crop writes an inference-only config.
from .misc import tensor2img

def psnr(*args, **kwargs):
    raise RuntimeError('PSNR is unavailable in Crop inference mode.')

def ssim(*args, **kwargs):
    raise RuntimeError('SSIM is unavailable in Crop inference mode.')

class InceptionV3:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('InceptionV3 is unavailable in Crop inference mode.')

__all__ = ['tensor2img', 'psnr', 'ssim', 'InceptionV3']
PY

cat > "${MMEDIT_SITE_PACKAGES}/models/__init__.py" <<'PY'
# Crop Colab compatibility patch.
# Import only the modules needed for RealBasicVSR inference. The original
# mmedit package imports every model family, including modules that require
# mmcv-full CUDA extensions unavailable on current Colab Python.
from .builder import build, build_backbone, build_component, build_loss, build_model
from .registry import BACKBONES, COMPONENTS, LOSSES, MODELS
from .backbones.sr_backbones.basicvsr_net import BasicVSRNet
from .backbones.sr_backbones.real_basicvsr_net import RealBasicVSRNet
from .restorers.real_basicvsr import RealBasicVSR

__all__ = [
    'build', 'build_backbone', 'build_component', 'build_loss', 'build_model',
    'BACKBONES', 'COMPONENTS', 'LOSSES', 'MODELS', 'BasicVSRNet',
    'RealBasicVSRNet', 'RealBasicVSR'
]
PY

cat > "${MMEDIT_SITE_PACKAGES}/models/restorers/__init__.py" <<'PY'
# Crop Colab compatibility patch.
from .real_basicvsr import RealBasicVSR

__all__ = ['RealBasicVSR']
PY

cat > "${MMEDIT_SITE_PACKAGES}/models/backbones/__init__.py" <<'PY'
# Crop Colab compatibility patch.
from .sr_backbones.basicvsr_net import BasicVSRNet
from .sr_backbones.real_basicvsr_net import RealBasicVSRNet

__all__ = ['BasicVSRNet', 'RealBasicVSRNet']
PY

cat > "${MMEDIT_SITE_PACKAGES}/models/backbones/sr_backbones/__init__.py" <<'PY'
# Crop Colab compatibility patch.
from .basicvsr_net import BasicVSRNet
from .real_basicvsr_net import RealBasicVSRNet

__all__ = ['BasicVSRNet', 'RealBasicVSRNet']
PY

mkdir -p "$(dirname "${CHECKPOINT_PATH}")"
if [ ! -s "${CHECKPOINT_PATH}" ]; then
    gdown "${CHECKPOINT_FILE_ID}" -O "${CHECKPOINT_PATH}"
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
