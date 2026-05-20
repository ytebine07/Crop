import os
import unittest
from unittest import mock

from modules.constants import Constants as Const
from modules.enhancer import FrameEnhancer, NoOpEnhancer


class TestFrameEnhancer(unittest.TestCase):
    def test_create_default_can_be_disabled(self):
        with mock.patch.dict(os.environ, {Const.ENHANCER_ENV: Const.ENHANCER_OFF}):
            with mock.patch("builtins.print") as print_mock:
                self.assertIsInstance(FrameEnhancer.create_default(), NoOpEnhancer)

        log = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Real-ESRGAN disabled", log)
        self.assertNotIn("Using real-esrgan", log)

    def test_create_default_falls_back_when_realesrgan_is_unavailable(self):
        with mock.patch.dict(os.environ, {Const.ENHANCER_ENV: Const.ENHANCER_AUTO}):
            with mock.patch(
                "modules.enhancer.RealESRGANEnhancer.create",
                side_effect=RuntimeError("missing dependency"),
            ):
                with mock.patch("builtins.print") as print_mock:
                    self.assertIsInstance(FrameEnhancer.create_default(), NoOpEnhancer)

        log = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Real-ESRGAN is unavailable", log)
        self.assertIn("Using standard-resize", log)

    def test_create_default_logs_when_realesrgan_is_used(self):
        enhancer = mock.Mock()
        enhancer.name = "real-esrgan"
        with mock.patch.dict(os.environ, {Const.ENHANCER_ENV: Const.ENHANCER_AUTO}):
            with mock.patch(
                "modules.enhancer.RealESRGANEnhancer.create",
                return_value=enhancer,
            ):
                with mock.patch("builtins.print") as print_mock:
                    self.assertIs(FrameEnhancer.create_default(), enhancer)

        log = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Using real-esrgan", log)
