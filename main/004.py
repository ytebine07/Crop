"""
004.py
colab用にリファクタリング
"""
from PIL import Image
import tqdm
import glob
import os
from imageai.Detection import ObjectDetection
import sys
from tqdm import tqdm
sys.path.append('/usr/local/lib/python3.6/lib-dynload')
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/local/lib/python36.zip')
sys.path.append('/usr/local/lib/python3.6')
execution_path = os.getcwd()

# 定義系

AVERAGE_FRAME = os.getenv("AVERAGE_FRAME", 3)
model_path = os.getenv(
    "MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
input_path = os.getenv("INPUT_PATH", "./data/input5/*png")
output_path = os.getenv("OUTPUT_PATH", "./data/output/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", "./data/output_croped/")  # cropされた画像の出力先


def main():

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()

    frame_count = 0
    is_human = False
    new_positions = []
    for file in tqdm(glob.glob(input_path)):
        input_image = os.path.join(execution_path, os.path.dirname(
            input_path), os.path.basename(file))
        output_image_path = os.path.join(
            execution_path, output_path, os.path.basename(file))
        # ここでオブジェクト判別 & 画像書き出し
        detections = detector.detectObjectsFromImage(
            input_image=input_image, output_image_path=output_image_path)

        # 複数のオブジェクトを検出する可能性があるのでforで回す
        for d in detections:
            if d["name"] == "person" and d["percentage_probability"] > 90:
                is_human = True
                x1 = d["box_points"][0]
                y1 = d["box_points"][1]
                x2 = d["box_points"][2]
                y2 = d["box_points"][3]
                center = (x1 + x2) // 2

                nx1 = center - (612//2)
                nx2 = center + (612//2)

                hoge = calcurate_new_position(
                    new_positions, nx1, nx2, center, frame_count)

                # 座標を保存
                # new_positions.append([nx1, nx2, center])
                new_positions.append(hoge)

        # フレーム中に人間を検出出来なかったら
        # 1コマ前に検出された座標を入れておく(nx1,nx2に入ったままになっている)
        if is_human == False:
            # new_positions.append([nx1, nx2, center])
            new_positions.append(hoge)

        # cropファイルの切り出し
        im = Image.open(file)
        # im_crop = im.crop((nx1, 0, nx2, 1088)).save(
        #    c_output_path + os.path.basename(file), quality=100)

        im_crop = im.crop((hoge[0], 0, hoge[1], 1088)).save(
            c_output_path + os.path.basename(file), quality=100)

        frame_count += 1
        is_human = False  # 初期化


def calcurate_new_position(new_positions, x1: int, x2: int, center: int, frame_count: int):

    # 現在のフレームから遡って計算するが、フレーム数がマイナスの場合は遡り先がない。
    # その場合は現在の座標を入れておく。
    if (frame_count-AVERAGE_FRAME) <= 0:
        return [x1, x2, center]

    # 新しいcenterを計算する
    new_centers = []
    saka = frame_count - AVERAGE_FRAME
    for i in range(saka, frame_count):
        new_centers.append(new_positions[i][2])

    s = sum(new_centers)
    n = len(new_centers)
    new_center = s//n

    # TODO 新しいnx1,nx2を計算する
    # TODO 左右はみ出したらはみ出させないようにする
    nx1 = new_center - (612//2)
    nx2 = new_center + (612//2)

    return [nx1, nx2, new_center]


if __name__ == '__main__':
    main()
