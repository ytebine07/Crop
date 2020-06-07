import unittest

from modules.video import Video
from modules.video_resource import VideoResource
from modules.encoder import Encoder


class Test_Encoder(unittest.TestCase):
    def test_hoge(self):
        baseDir = "/tmp/test"
        video = Video("10.mp4")
        vr = VideoResource(video, baseDir)

        Encoder(baseDir, vr.get_image_path(), vr.get_sound_path(), video.fps).encode()
