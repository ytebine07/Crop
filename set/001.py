"""
001.py
検出したオブジェクトに枠を付けて画像出力する
"""

from imageai.Detection import ObjectDetection
import os
import glob

import tqdm

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
    print(eachObject["name"], " : ", eachObject["percentage_probability"])
