import os


class Constants:
    ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(ROOT_DIR_PATH, "model")
    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "tiny-yolov3.pt")
    REAL_ESRGAN_MODEL_DIR = os.path.join(MODEL_PATH, "realesrgan")
    REAL_ESRGAN_MODEL_NAME = "RealESRGAN_x4plus"
    REAL_ESRGAN_MODEL_URL = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth"
    )

    AVERAGE_FLAMES = 120
    DETECT_INTERVAL_FRAMES = 10

    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920

    VIDEO_CRF = 18
    VIDEO_PRESET = "slow"
    VIDEO_PROFILE = "high"

    ENHANCER_ENV = "CROP_ENHANCER"
    ENHANCER_AUTO = "auto"
    ENHANCER_OFF = "off"
    REAL_ESRGAN_OUTSCALE = 2
    REAL_ESRGAN_TILE = 0
    REAL_ESRGAN_TILE_PAD = 10
    REAL_ESRGAN_PRE_PAD = 0
    CROP_PROGRESS_LOG_INTERVAL_FRAMES = 10
