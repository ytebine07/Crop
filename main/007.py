"""
007.py
フォルダ構成変更
"""
import tensorflow as tf
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

tf.logging.set_verbosity(tf.logging.ERROR)

# 定義系

AVERAGE_FRAME = int(os.getenv("AVERAGE_FRAME", 30))
QUALITY = os.getenv("QUALITY", 'fastest')

model_path = os.getenv(
    "MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
base_dir = os.getenv("BASE_DIR", "./data/set/")
input_path = os.getenv("INPUT_PATH", base_dir + "/set/images/*png")
output_path = os.getenv("OUTPUT_PATH", base_dir + "./image2/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", base_dir + "/set/croped/")  # cropされた画像の出力先
positions_path = os.getenv("POSITIONS_PATH", base_dir + "/positions.txt")


def main():

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed=QUALITY)

    print("-----------------------------------------------------")
    print("AVERAGE FLAME : " + str(AVERAGE_FRAME))
    print("QUALITY       : " + QUALITY)
    print("-----------------------------------------------------")

    frame_count = 0
    is_human = False
    original_centers = []
    center = 960
    for file in tqdm(sorted(glob.glob(input_path)), desc="画像解析"):
        input_image = os.path.join(execution_path, os.path.dirname(
            input_path), os.path.basename(file))
        output_image_path = os.path.join(
            execution_path, output_path, os.path.basename(file))

        # ここでオブジェクト判別
        detections = detector.detectObjectsFromImage(
            input_image=input_image, output_image_path=output_image_path, output_type='array')

        # 複数のオブジェクトを検出する可能性があるのでforで回す
        for d in detections[1]:
            if d["name"] == "person" and d["percentage_probability"] > 80:
                is_human = True
                x1 = d["box_points"][0]
                x2 = d["box_points"][2]
                center = (x1 + x2) // 2

                # 座標を保存
                original_centers.append(center)
            else:
                pass
                # print("nohuman frame : " + str(frame_count))
                # per = d["name"] == "persion"
                # ritu = d["percentage_probability"] > 80

        # フレーム中に人間を検出出来なかったら、1コマ前に検出された座標を入れておく
        if is_human == False:
            original_centers.append(center)

        frame_count += 1
        is_human = False

    f = open(positions_path, 'w')
    f_count = 1
    for line in original_centers:
        a = "{0},{1}\n".format(f_count, int(line))
        f.write(a)
        f_count += 1
    f.close

    # 移動平均の算出
    conv = Convolve(AVERAGE_FRAME, original_centers)
    y3 = conv.calculate()

    fc = 0
    for file in tqdm(sorted(glob.glob(input_path)), desc="ファイル書き出し"):
        croped_output_image_path = os.path.join(execution_path,
                                                c_output_path,
                                                os.path.basename(file))

        n1 = y3[fc] - (612//2)
        n2 = y3[fc] + (612//2)

        # ここで画像書き出し
        Image.open(file).crop((n1, 0, n2, 1080)).save(
            croped_output_image_path, quality=100)
        fc += 1


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import Convolve, Common

    main()
