import os
import importlib
import subprocess
import sys

from modules.constants import Constants as Const
from modules.dir import Directory


def install_torchvision_functional_tensor_compatibility():
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return False

    try:
        importlib.import_module(module_name)
        return False
    except ModuleNotFoundError:
        pass

    for replacement_name in (
        "torchvision.transforms._functional_tensor",
        "torchvision.transforms.functional",
    ):
        try:
            replacement = importlib.import_module(replacement_name)
            sys.modules[module_name] = replacement
            print(
                "[Enhancer] Installed torchvision compatibility shim: {0} -> {1}".format(
                    module_name,
                    replacement_name,
                )
            )
            return True
        except ModuleNotFoundError:
            continue

    raise ModuleNotFoundError("No module named '{0}'".format(module_name))


def get_cuda_diagnostics(torch):
    diagnostics = [
        "torch.__version__={0}".format(getattr(torch, "__version__", "unknown")),
        "torch.version.cuda={0}".format(getattr(torch.version, "cuda", None)),
    ]
    try:
        diagnostics.append("torch.cuda.device_count()={0}".format(torch.cuda.device_count()))
    except Exception as error:
        diagnostics.append("torch.cuda.device_count() failed: {0}".format(error))

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_names = ", ".join(
                line.strip() for line in result.stdout.splitlines() if line.strip()
            )
            diagnostics.append("nvidia-smi detected GPU(s): {0}".format(gpu_names))
        else:
            message = result.stderr.strip() or result.stdout.strip()
            diagnostics.append("nvidia-smi failed: {0}".format(message))
    except Exception as error:
        diagnostics.append("nvidia-smi unavailable: {0}".format(error))

    return diagnostics


def ensure_cuda_available(torch):
    if torch.cuda.is_available():
        try:
            print("[Enhancer] CUDA available: {0}".format(torch.cuda.get_device_name(0)))
        except Exception:
            print("[Enhancer] CUDA available.")
        return

    for diagnostic in get_cuda_diagnostics(torch):
        print("[Enhancer] CUDA diagnostic: {0}".format(diagnostic))

    raise RuntimeError(
        "CUDA is not available. In Google Colab, select a GPU runtime and make sure "
        "the installed PyTorch build includes CUDA support."
    )


class NoOpEnhancer:
    name = "standard-resize"

    def enhance(self, frame):
        return frame


class RealESRGANEnhancer:
    name = "real-esrgan"

    def __init__(self, upsampler):
        self.__upsampler = upsampler

    @classmethod
    def create(cls):
        import torch

        ensure_cuda_available(torch)
        install_torchvision_functional_tensor_compatibility()

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer

        Directory.create(Const.REAL_ESRGAN_MODEL_DIR)
        model_path = os.path.join(
            Const.REAL_ESRGAN_MODEL_DIR, Const.REAL_ESRGAN_MODEL_NAME + ".pth"
        )
        if not os.path.exists(model_path):
            model_path = load_file_from_url(
                url=Const.REAL_ESRGAN_MODEL_URL,
                model_dir=Const.REAL_ESRGAN_MODEL_DIR,
                progress=True,
                file_name=Const.REAL_ESRGAN_MODEL_NAME + ".pth",
            )

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=Const.REAL_ESRGAN_TILE,
            tile_pad=Const.REAL_ESRGAN_TILE_PAD,
            pre_pad=Const.REAL_ESRGAN_PRE_PAD,
            half=True,
        )
        print("[Enhancer] Real-ESRGAN enabled. model={0}, tile={1}".format(
            Const.REAL_ESRGAN_MODEL_NAME,
            Const.REAL_ESRGAN_TILE,
        ))
        return cls(upsampler)

    def enhance(self, frame):
        enhanced_frame, _ = self.__upsampler.enhance(
            frame, outscale=Const.REAL_ESRGAN_OUTSCALE
        )
        return enhanced_frame


class FrameEnhancer:
    @staticmethod
    def create_default():
        mode = os.environ.get(Const.ENHANCER_ENV, Const.ENHANCER_AUTO).lower()
        if mode in (Const.ENHANCER_OFF, "0", "false", "none"):
            print("[Enhancer] Real-ESRGAN disabled by {0}={1}.".format(
                Const.ENHANCER_ENV,
                mode,
            ))
            return NoOpEnhancer()

        try:
            enhancer = RealESRGANEnhancer.create()
            print("[Enhancer] Using {0}.".format(enhancer.name))
            return enhancer
        except Exception as error:
            print("[Enhancer] Real-ESRGAN is unavailable. Fallback to standard resize.")
            print("[Enhancer] {0}".format(error))
            enhancer = NoOpEnhancer()
            print("[Enhancer] Using {0}.".format(enhancer.name))
            return enhancer
