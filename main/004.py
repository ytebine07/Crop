"""
004.py
colab用にリファクタリング
"""
from PIL import Image
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

AVERAGE_FRAME = int(os.getenv("AVERAGE_FRAME", 15))
model_path = os.getenv(
    "MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
input_path = os.getenv("INPUT_PATH", "./data/input5/*png")
output_path = os.getenv("OUTPUT_PATH", "./data/output/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", "./data/output_croped")  # cropされた画像の出力先
SLACK = os.getenv("SLACK", False)


def main():

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel(detection_speed='fastest')

    frame_count = 0
    is_human = False
    positions = []
    for file in tqdm(glob.glob(input_path)):
        input_image = os.path.join(execution_path, os.path.dirname(
            input_path), os.path.basename(file))
        output_image_path = os.path.join(
            execution_path, output_path, os.path.basename(file))
        croped_output_image_path = os.path.join(execution_path,
                                                c_output_path, os.path.basename(file))

        # ここでオブジェクト判別
        detections = detector.detectObjectsFromImage(
            input_image=input_image, output_image_path=output_image_path, output_type='array')

        # 複数のオブジェクトを検出する可能性があるのでforで回す
        for d in detections[1]:
            if d["name"] == "person" and d["percentage_probability"] > 80:
                is_human = True
                x1 = d["box_points"][0]
                y1 = d["box_points"][1]
                x2 = d["box_points"][2]
                y2 = d["box_points"][3]
                center = (x1 + x2) // 2

                nx1 = center - (612//2)
                nx2 = center + (612//2)

                new_position = calcurate_new_position(
                    positions, nx1, nx2, center, frame_count)

                # 座標を保存
                positions.append(new_position)

        # フレーム中に人間を検出出来なかったら
        # 1コマ前に検出された座標を入れておく(nx1,nx2に入ったままになっている)
        if is_human == False:
            positions.append(new_position)

        # cropファイルの切り出し
        Image.open(file).crop((new_position[0], 0, new_position[1], 1088)).save(
            croped_output_image_path, quality=100)

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


def slack_notify(msg='おわったよ'):
    if SLACK:
        slack_user_id = os.getenv("SLACK_USER_ID", "")
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        requests.post(slack_webhook_url, json={
            "text": f"<@{slack_user_id}> {msg}"})


if __name__ == '__main__':
    main()
    slack_notify()
