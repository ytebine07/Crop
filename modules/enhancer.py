import os

from modules.constants import Constants as Const
from modules.dir import Directory


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
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

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
