import numpy as np
from typing import List, Optional
from imageai.Detection import ObjectDetection


from modules.constants import Constants as const
from modules.video import Video


class Person:
    def __init__(self, detection):

        x1 = detection["box_points"][0]
        y1 = detection["box_points"][1]
        x2 = detection["box_points"][2]
        y2 = detection["box_points"][3]

        self.center_x: int = (x1 + x2) // 2
        self.center_y: int = (y1 + y2) // 2
        self.median: np.array = np.array([self.center_x, self.center_y])
        self.distance = 99999


class ActorDetector:
    def __init__(self, screen: Video):
        self.__detector = ObjectDetection()
        self.__detector.setModelTypeAsRetinaNet()
        self.__detector.setModelPath(const.MODEL_FILE_PATH)
        self.__detector.loadModel(detection_speed="fastest")

        self.__screen_x: int = screen.width // 2
        self.__screen_y: int = screen.height // 2
        self.__screen_median: np.array = np.array([self.__screen_x, self.__screen_y])

    def get_actor(self, imagePath: str) -> Optional[Person]:
        detections = self.__detector.detectObjectsFromImage(
            input_image=imagePath, output_type="array"
        )

        persons = self.__extract_persons(detections)
        if 0 == len(persons):
            return None

        return self.__extract_actor(persons)

    def __extract_persons(self, detections) -> List[Person]:
        persons_list = []
        for d in detections[1]:
            if self.__is_person(d):
                persons_list.append(Person(d))
        return persons_list

    def __extract_actor(self, persons: List[Person]) -> Optional[Person]:
        actor = None
        for person in persons:
            # 画面の中心点に最も近いpersonをactorとする
            distance = np.linalg.norm(self.__screen_median - person.median)
            person.distance = distance
            if (actor is None) or (actor.distance > person.distance):
                actor = person
        return actor

    def __is_person(self, detection) -> bool:
        if detection["name"] == "person" and detection["percentage_probability"] > 30:
            return True
        return False
