"""
002.py
オブジェクトの座標を取得する
"""
from PIL import Image
import tqdm
import glob
import os
from imageai.Detection import ObjectDetection
import sys
sys.path.append('/usr/local/lib/python3.6/lib-dynload')
sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/local/lib/python36.zip')
sys.path.append('/usr/local/lib/python3.6')

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
    execution_path, "../model/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = []
for file in glob.glob("./input/*.png"):
    print(os.path.basename(file))
    detections.append(detector.detectObjectsFromImage(input_image=os.path.join(
        execution_path, "./input/" + os.path.basename(file)), output_image_path=os.path.join(execution_path, "./output/" + os.path.basename(file))))

for eachObject in detections:
    for ea in eachObject:
        if ea["name"] == 'person' and ea["percentage_probability"] > 90:
            x1 = ea["box_points"][0]
            y1 = ea["box_points"][1]
            x2 = ea["box_points"][2]
            y2 = ea["box_points"][3]
            center = (x1 + x2) // 2  # 切り捨て
