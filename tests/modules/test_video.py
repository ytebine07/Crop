import unittest

from modules.video import Video


class TestVideo(unittest.TestCase):
    # def test_exists_video(self):
    #     v = Video("sample.mp4")
    #     print(v.path)
    #     print(v.height)
    #     print(v.width)
    #     print(v.fps)

    def test_no_video(self):
        with self.assertRaises(Exception):
            Video("nofile.mp4")
