import unittest

from modules.video import Video
from modules.actor_detector import ActorDetector


class TestActorDetector(unittest.TestCase):
    # def test_detect(self):
    #     ad = ActorDetector(Video("./10.mp4"))
    #     ad.detect("/tmp/hoge/image_00001.png")

    def test_get_actor(self):
        ad = ActorDetector(Video("./10.mp4"))
        ad.get_actor("/tmp/hoge/image_00001.png")
