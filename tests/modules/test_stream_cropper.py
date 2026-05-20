import unittest
from types import SimpleNamespace
from unittest import mock

from modules.constants import Constants as Const
from modules.stream_cropper import StreamCropper


class FakeFrame:
    def __init__(self):
        self.slices = []

    def __getitem__(self, key):
        self.slices.append(key)
        return self


class FakeImage:
    def __init__(self, width, height):
        self.shape = (height, width, 3)


class FakeCv2:
    INTER_AREA = 3
    INTER_LANCZOS4 = 4

    def __init__(self):
        self.resize_calls = []

    def resize(self, frame, size, interpolation):
        self.resize_calls.append((frame, size, interpolation))
        return frame


class TestStreamCropper(unittest.TestCase):
    def test_crop_frame_clamps_left_edge(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())
        frame = FakeFrame()

        cropper._StreamCropper__crop_frame(frame, 0)

        self.assertEqual(frame.slices[0], (slice(0, 1080, None), slice(0, 607, None)))

    def test_resize_uses_lanczos_when_upscaling(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())
        cv2 = FakeCv2()

        cropper._StreamCropper__resize_frame(FakeImage(607, 1080), cv2)

        self.assertEqual(cv2.resize_calls[0][1], (Const.OUTPUT_WIDTH, Const.OUTPUT_HEIGHT))
        self.assertEqual(cv2.resize_calls[0][2], FakeCv2.INTER_LANCZOS4)

    def test_resize_uses_area_when_downscaling(self):
        video = SimpleNamespace(width=7680, height=4320, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())
        cv2 = FakeCv2()

        cropper._StreamCropper__resize_frame(FakeImage(2160, 3840), cv2)

        self.assertEqual(cv2.resize_calls[0][2], FakeCv2.INTER_AREA)

    def test_ffmpeg_command_uses_high_quality_settings(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())

        with mock.patch("subprocess.Popen") as popen:
            cropper._StreamCropper__create_ffmpeg_process()

        command = popen.call_args[0][0]
        self.assertIn("-crf", command)
        self.assertIn(str(Const.VIDEO_CRF), command)
        self.assertIn("-preset", command)
        self.assertIn(Const.VIDEO_PRESET, command)
        self.assertIn("{0}x{1}".format(Const.OUTPUT_WIDTH, Const.OUTPUT_HEIGHT), command)
