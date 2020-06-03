import unittest

from modules.video import Video
from modules.video_resource import VideoResource


class TestVideoResource(unittest.TestCase):
    # def test_hoge(self):
    #    video = Video("./10.mp4")
    #    VideoResource(video, "/tmp/hoge/").create()

    def test_hoge2(self):
        video = Video("./10.mp4")
        vr = VideoResource(video, "/tmp/hoge")
        for r in vr.get_image_paths():
            print(r)
        print(vr.get_sound_path())
