import os


class Constants:
    ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(ROOT_DIR_PATH, "model")
    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "tiny-yolov3.pt")

    AVERAGE_FLAMES = 120
