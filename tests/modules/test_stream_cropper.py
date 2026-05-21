import unittest
import sys
from types import SimpleNamespace
from unittest import mock

from modules.constants import Constants as Const
from modules.stream_cropper import StreamCropper


class FakeFrame:
    def __init__(self):
        self.slices = []
        self.shape = (1080, 1920, 3)

    def __getitem__(self, key):
        self.slices.append(key)
        return self

    def tobytes(self):
        return b"frame"


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


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, _):
        return len(self.frames)

    def read(self):
        if len(self.frames) == 0:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeProcess:
    def __init__(self):
        self.stdin = mock.Mock()

    def wait(self):
        return 0


class FakeProgress:
    def __init__(self, total=None, desc=None, unit=None, **_):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, amount):
        self.updates.append(amount)


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

    def test_crop_reports_frame_progress(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        enhancer = SimpleNamespace(enhance=lambda frame: frame)
        cropper = StreamCropper(video, "output.mp4", enhancer=enhancer)
        capture = FakeCapture([FakeFrame(), FakeFrame()])
        process = FakeProcess()
        progress_instances = []

        def create_progress(*args, **kwargs):
            progress = FakeProgress(*args, **kwargs)
            progress_instances.append(progress)
            return progress

        fake_cv2 = SimpleNamespace(
            VideoCapture=mock.Mock(return_value=capture),
            CAP_PROP_FRAME_COUNT=7,
            INTER_AREA=FakeCv2.INTER_AREA,
            INTER_LANCZOS4=FakeCv2.INTER_LANCZOS4,
            resize=FakeCv2().resize,
        )
        fake_tqdm = SimpleNamespace(tqdm=create_progress)

        with mock.patch.dict(sys.modules, {"cv2": fake_cv2, "tqdm": fake_tqdm}):
            with mock.patch.object(cropper, "_StreamCropper__create_ffmpeg_process", return_value=process):
                cropper.crop([100, 120], total_frames=2)

        self.assertEqual(progress_instances[0].total, 2)
        self.assertEqual(progress_instances[0].desc, "Crop/Enhance frames")
        self.assertEqual(progress_instances[0].unit, "frame")
        self.assertEqual(progress_instances[0].updates, [1, 1])

    def test_print_frame_progress_logs_first_interval_and_last_frame(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())

        with mock.patch("modules.stream_cropper.time.perf_counter", return_value=20):
            with mock.patch("builtins.print") as print_mock:
                cropper._StreamCropper__print_frame_progress(1, 25, 0)
                cropper._StreamCropper__print_frame_progress(2, 25, 0)
                cropper._StreamCropper__print_frame_progress(10, 25, 0)
                cropper._StreamCropper__print_frame_progress(25, 25, 0)

        logs = [call.args[0] for call in print_mock.call_args_list]
        self.assertEqual(len(logs), 3)
        self.assertIn("1/25", logs[0])
        self.assertIn("10/25", logs[1])
        self.assertIn("25/25", logs[2])

    def test_format_seconds(self):
        video = SimpleNamespace(width=1920, height=1080, fps=30, path="input.mp4")
        cropper = StreamCropper(video, "output.mp4", enhancer=SimpleNamespace())

        self.assertEqual(cropper._StreamCropper__format_seconds(8), "8s")
        self.assertEqual(cropper._StreamCropper__format_seconds(68), "1m08s")
        self.assertEqual(cropper._StreamCropper__format_seconds(3668), "1h01m08s")
