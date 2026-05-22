import os
import tempfile
import unittest
from unittest import mock

from modules.constants import Constants as Const
from modules.video_enhancer import (
    NoOpVideoEnhancer,
    RealBasicVSREnhancer,
    VideoEnhancer,
)


class TestVideoEnhancer(unittest.TestCase):
    def test_is_requested_when_realbasicvsr_is_selected(self):
        with mock.patch.dict(os.environ, {Const.ENHANCER_ENV: Const.ENHANCER_REAL_BASIC_VSR}):
            self.assertTrue(VideoEnhancer.is_requested())

    def test_is_not_requested_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(VideoEnhancer.is_requested())

    def test_create_default_returns_realbasicvsr(self):
        with mock.patch.dict(os.environ, {Const.ENHANCER_ENV: Const.ENHANCER_REAL_BASIC_VSR}):
            enhancer = VideoEnhancer.create_default("/tmp/work", 30)

        self.assertIsInstance(enhancer, RealBasicVSREnhancer)

    def test_noop_copies_video_when_paths_differ(self):
        with mock.patch("modules.video_enhancer.subprocess.run") as run_mock:
            NoOpVideoEnhancer().enhance("input.mp4", "output.mp4")

        self.assertEqual(
            run_mock.call_args[0][0],
            ["ffmpeg", "-y", "-i", "input.mp4", "-c", "copy", "output.mp4"],
        )

    def test_noop_validate_succeeds(self):
        self.assertIsInstance(NoOpVideoEnhancer().validate(), NoOpVideoEnhancer)

    def test_realbasicvsr_runs_extract_inference_and_encode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo_dir = os.path.join(tempdir, "RealBasicVSR")
            config_dir = os.path.join(repo_dir, "configs")
            checkpoint_path = os.path.join(repo_dir, "weights", "model.pth")
            os.makedirs(config_dir)
            os.makedirs(os.path.dirname(checkpoint_path))
            script_path = os.path.join(repo_dir, "inference_realbasicvsr.py")
            config_path = os.path.join(config_dir, "config.py")
            for path in (script_path, config_path, checkpoint_path):
                with open(path, "w") as file:
                    file.write("placeholder")

            enhancer = RealBasicVSREnhancer(
                workdir=tempdir,
                fps=29.97,
                repo_dir=repo_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                max_seq_len=12,
            )

            process = mock.Mock()
            process.poll.return_value = 0
            with mock.patch("modules.video_enhancer.glob.glob", return_value=["00000000.png"]):
                with mock.patch("modules.video_enhancer.subprocess.run") as run_mock:
                    with mock.patch(
                        "modules.video_enhancer.subprocess.Popen",
                        return_value=process,
                    ) as popen_mock:
                        enhancer.enhance("cropped_raw.mp4", "nosound.mp4")

            commands = [call.args[0] for call in run_mock.call_args_list]
            self.assertEqual(commands[0][0:4], ["ffmpeg", "-y", "-i", "cropped_raw.mp4"])
            inference_command = popen_mock.call_args.args[0]
            self.assertEqual(
                inference_command[0:4],
                ["python", script_path, config_path, checkpoint_path],
            )
            self.assertIn("--max_seq_len=12", inference_command)
            self.assertIn("--is_save_as_png=True", inference_command)
            self.assertIn("--fps=29.97", inference_command)
            inference_env = popen_mock.call_args.kwargs["env"]
            compat_path = os.path.join(tempdir, "realbasicvsr_py_compat")
            self.assertTrue(inference_env["PYTHONPATH"].startswith(compat_path))
            self.assertEqual(inference_env["PYTHONUNBUFFERED"], "1")
            with open(os.path.join(compat_path, "sitecustomize.py")) as sitecustomize:
                sitecustomize_content = sitecustomize.read()
            self.assertIn("ImpImporter", sitecustomize_content)
            self.assertIn("'bool': bool", sitecustomize_content)
            self.assertIn("'bool8': _np.bool_", sitecustomize_content)
            self.assertIn("if _name not in _np.__dict__", sitecustomize_content)
            self.assertEqual(commands[1][0:4], ["ffmpeg", "-y", "-framerate", "29.97"])
            self.assertIn("-crf", commands[1])
            self.assertIn(str(Const.VIDEO_CRF), commands[1])
            self.assertEqual(commands[1][-1], "nosound.mp4")

    def test_realbasicvsr_logs_inference_progress(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo_dir = os.path.join(tempdir, "RealBasicVSR")
            config_dir = os.path.join(repo_dir, "configs")
            checkpoint_path = os.path.join(repo_dir, "weights", "model.pth")
            os.makedirs(config_dir)
            os.makedirs(os.path.dirname(checkpoint_path))
            script_path = os.path.join(repo_dir, "inference_realbasicvsr.py")
            config_path = os.path.join(config_dir, "config.py")
            for path in (script_path, config_path, checkpoint_path):
                with open(path, "w") as file:
                    file.write("placeholder")

            enhancer = RealBasicVSREnhancer(
                workdir=tempdir,
                fps=30,
                repo_dir=repo_dir,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                progress_interval_seconds=1,
            )

            process = mock.Mock()
            process.poll.side_effect = [None, 0]

            def fake_glob(pattern):
                if "realbasicvsr_input" in pattern:
                    return ["input_1.png", "input_2.png"]
                if "realbasicvsr_output" in pattern:
                    return ["output_1.png"]
                return []

            with mock.patch("modules.video_enhancer.glob.glob", side_effect=fake_glob):
                with mock.patch("modules.video_enhancer.subprocess.run"):
                    with mock.patch("modules.video_enhancer.subprocess.Popen", return_value=process):
                        with mock.patch("modules.video_enhancer.time.sleep"):
                            with mock.patch("modules.video_enhancer.time.monotonic", side_effect=[0, 0, 1]):
                                with mock.patch("builtins.print") as print_mock:
                                    enhancer.enhance("cropped_raw.mp4", "nosound.mp4")

            logs = "\n".join(call.args[0] for call in print_mock.call_args_list)
            self.assertIn("RealBasicVSR inference: start", logs)
            self.assertIn("RealBasicVSR inference: 1/2 (50.0%)", logs)
            self.assertIn("RealBasicVSR inference: finished", logs)

    def test_realbasicvsr_fails_when_repo_is_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            enhancer = RealBasicVSREnhancer(
                workdir=tempdir,
                fps=30,
                repo_dir=os.path.join(tempdir, "missing"),
                config_path="config.py",
                checkpoint_path=os.path.join(tempdir, "missing.pth"),
            )

            with self.assertRaisesRegex(RuntimeError, "RealBasicVSR is not ready"):
                enhancer.enhance("cropped_raw.mp4", "nosound.mp4")

    def test_realbasicvsr_validate_fails_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tempdir:
            enhancer = RealBasicVSREnhancer(
                workdir=tempdir,
                fps=30,
                repo_dir=os.path.join(tempdir, "missing"),
                config_path="config.py",
                checkpoint_path=os.path.join(tempdir, "missing.pth"),
            )

            with mock.patch("modules.video_enhancer.subprocess.run") as run_mock:
                with self.assertRaisesRegex(RuntimeError, Const.REAL_BASIC_VSR_REPO_ENV):
                    enhancer.validate()

        run_mock.assert_not_called()
