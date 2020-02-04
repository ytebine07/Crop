"""
005.py
座標算出に移動平均を使用
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

AVERAGE_FRAME = int(os.getenv("AVERAGE_FRAME", 5))
model_path = os.getenv(
    "MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
input_path = os.getenv("INPUT_PATH", "./data/input5/*png")
output_path = os.getenv("OUTPUT_PATH", "./data/output/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", "./data/output_croped")  # cropされた画像の出力先
positions_path = os.getenv("POSITIONS_PATH", "./positions.txt")
SLACK = os.getenv("SLACK", False)


def main():

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    # detector.loadModel(detection_speed='normal')
    detector.loadModel(detection_speed='fastest')

    frame_count = 0
    is_human = False
    positions = []
    raw_centers = []
    for file in tqdm(glob.glob(input_path), desc="画像解析"):
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
                raw_centers.append(center)
            else:
                pass
                #print("nohuman frame : " + str(frame_count))
                #per = d["name"] == "persion"
                #ritu = d["percentage_probability"] > 80

        # フレーム中に人間を検出出来なかったら
        # 1コマ前に検出された座標を入れておく(nx1,nx2に入ったままになっている)
        if is_human == False:
            raw_centers.append(center)

        frame_count += 1
        is_human = False  # 初期化

    # 移動平均の算出
    # @see https://deepage.net/features/numpy-convolve.html
    v = np.ones(AVERAGE_FRAME)/AVERAGE_FRAME
    y3 = np.convolve(raw_centers, v, mode='valid')

    fc = 0
    for file in tqdm(glob.glob(input_path), desc="ファイル書き出し"):
        croped_output_image_path = os.path.join(execution_path,
                                                c_output_path, os.path.basename(file))

        # フレーム数に対して移動平均は足りないので、帳尻合わせる
        if fc >= len(y3):
            c = int(y3[len(y3)-1])
        else:
            c = int(y3[fc])

        n1 = c - (612//2)
        n2 = c + (612//2)

        # ここで書き出し
        Image.open(file).crop((n1, 0, n2, 1088)).save(
            croped_output_image_path, quality=100)
        fc += 1

    f = open(positions_path, 'w')
    f_count = 1
    for line in positions:
        a = "{0},{1},{2},{3}\n".format(f_count, line[0], line[1], line[2])
        f.write(a)
        f_count += 1
    f.close


if __name__ == '__main__':
    main()
