import os


class Constants:
    ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(ROOT_DIR_PATH, "model")
    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "tiny-yolov3.pt")
    REAL_ESRGAN_MODEL_DIR = os.path.join(MODEL_PATH, "realesrgan")
    REAL_ESRGAN_MODEL_NAME = "realesr-general-x4v3"
    REAL_ESRGAN_MODEL_URLS = {
        "RealESRGAN_x4plus": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
            "RealESRGAN_x4plus.pth",
        ],
        "realesr-general-x4v3": [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
            "realesr-general-x4v3.pth",
        ],
    }

    AVERAGE_FLAMES = 120
    DETECT_INTERVAL_FRAMES = 10

    OUTPUT_WIDTH = 1080
    OUTPUT_HEIGHT = 1920

    VIDEO_CRF = 16
    VIDEO_PRESET = "slow"
    VIDEO_PROFILE = "high"

    ENHANCER_ENV = "CROP_ENHANCER"
    ENHANCER_AUTO = "auto"
    ENHANCER_OFF = "off"
    ENHANCER_REAL_BASIC_VSR = "realbasicvsr"

    CROPPED_RAW_FILENAME = "cropped_raw.mp4"

    REAL_BASIC_VSR_REPO_ENV = "REAL_BASIC_VSR_REPO"
    REAL_BASIC_VSR_CONFIG_ENV = "REAL_BASIC_VSR_CONFIG"
    REAL_BASIC_VSR_CHECKPOINT_ENV = "REAL_BASIC_VSR_CHECKPOINT"
    REAL_BASIC_VSR_MAX_SEQ_LEN_ENV = "REAL_BASIC_VSR_MAX_SEQ_LEN"
    REAL_BASIC_VSR_REPO_DIR = "/content/RealBasicVSR"
    REAL_BASIC_VSR_CONFIG = "configs/realbasicvsr_x4.py"
    REAL_BASIC_VSR_CHECKPOINT = "/content/RealBasicVSR/checkpoints/RealBasicVSR_x4.pth"
    REAL_BASIC_VSR_MAX_SEQ_LEN = 5
    REAL_BASIC_VSR_PROGRESS_INTERVAL_SECONDS = 10

    REAL_ESRGAN_OUTSCALE = 4
    REAL_ESRGAN_TILE = 0
    REAL_ESRGAN_TILE_PAD = 10
    REAL_ESRGAN_PRE_PAD = 0
    REAL_ESRGAN_DENOISE_STRENGTH = 0.7
    CROP_PROGRESS_LOG_INTERVAL_FRAMES = 10

    UNSHARP_MASK_ENABLED = True
    UNSHARP_MASK_SIGMA = 1.0
    UNSHARP_MASK_AMOUNT = 0.25
