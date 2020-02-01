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
model_path = os.getenv("MODEL_PATH", "./model/resnet50_coco_best_v2.0.1.h5")
input_path = os.getenv("INPUT_PATH", "./data/input5/*png")
output_path = os.getenv("OUTPUT_PATH", "./data/output/")
c_output_path = os.getenv(
    "C_OUTPUT_PATH", "./data/output_croped/")  # cropされた画像の出力先

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

for file in tqdm(glob.glob(input_path)):
    input_image = os.path.join(execution_path, os.path.dirname(
        input_path), os.path.basename(file))
    output_image_path = os.path.join(
        execution_path, output_path, os.path.basename(file))
    # ここでオブジェクト判別 & 画像書き出し
    detections = detector.detectObjectsFromImage(
        input_image=input_image, output_image_path=output_image_path)

    for d in detections:
        if d["name"] == "person" and d["percentage_probability"] > 90:
            x1 = d["box_points"][0]
            y1 = d["box_points"][1]
            x2 = d["box_points"][2]
            y2 = d["box_points"][3]
            center = (x1 + x2) // 2

            nx1 = center - (612/2)
            nx2 = center + (612/2)

            im = Image.open(file)
            im_crop = im.crop((nx1, 0, nx2, 1088)).save(
                c_output_path + os.path.basename(file), quality=100)
