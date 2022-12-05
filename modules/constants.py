import os


class Constants:
    ROOT_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    MODEL_PATH = os.path.join(ROOT_DIR_PATH, "model")
    MODEL_FILE_PATH = os.path.join(MODEL_PATH, "resnet50_coco_best_v2.1.0.h5")

    AVERAGE_FLAMES = 120
