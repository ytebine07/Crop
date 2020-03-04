"""
008.py
ファイルから座標情報を利用する
"""
from PIL import Image
import numpy as np
import tqdm
import glob
import os
import requests
from imageai.Detection import ObjectDetection
import sys
from tqdm import tqdm
sys.path.append('/usr/local/lib/python3.6/lib-dynload')
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/local/lib/python36.zip')
sys.path.append('/usr/local/lib/python3.6')
execution_path = os.getcwd()

# 定義系

AVERAGE_FRAME = int(os.getenv("AVERAGE_FRAME", 30))
QUALITY = os.getenv("QUALITY", 'fastest')

model_path = os.getenv(
    "MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
# input_path = os.getenv("INPUT_PATH", "./data/input/*png")
# output_path = os.getenv("OUTPUT_PATH", "./data/output/")
# c_output_path = os.getenv(
#     "C_OUTPUT_PATH", "./data/output_croped")  # cropされた画像の出力先
# positions_path = os.getenv("POSITIONS_PATH", "./positions.txt")
SLACK = os.getenv("SLACK", False)
file_path = "./positions_fastest.txt"

base_dir = os.getenv("BASE_DIR", "./data/set6/")
input_path = os.getenv("INPUT_PATH", base_dir + "./image/*png")
output_path = os.getenv("OUTPUT_PATH", base_dir + "./image2/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", base_dir + "./croped/")  # cropされた画像の出力先
positions_path = os.getenv("POSITIONS_PATH", base_dir + "./positions.txt")


def main():

    print("--------------------------------")
    print("AVERAGE_FLAME : " + str(AVERAGE_FRAME))
    print("input file    : " + file_path)
    print("--------------------------------")

    common = Common()
    x, y = common.get_x_y(file_path)

    conv = Convolve(AVERAGE_FRAME, y)
    y = conv.calculate()

    fc = 0
    for file in tqdm(glob.glob(input_path), desc="ファイル書き出し"):
        croped_output_image_path = os.path.join(execution_path,
                                                c_output_path, os.path.basename(file))

        n1 = y[fc] - (612//2)
        n2 = y[fc] + (612//2)

        # ここで画像ファイルをcrop
        Image.open(file).crop((n1, 0, n2, 1080)).save(
            croped_output_image_path, quality=100)
        fc += 1


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import Convolve, Common

    main()
