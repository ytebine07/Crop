import importlib
import os
import sys
import types
import unittest
from unittest import mock

from modules.constants import Constants as Const
from modules.enhancer import (
    FrameEnhancer,
    NoOpEnhancer,
    install_torchvision_functional_tensor_compatibility,
)


TORCHVISION_MODULES = (
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional_tensor",
    "torchvision.transforms._functional_tensor",
    "torchvision.transforms.functional",
)
ORIGINAL_IMPORT_MODULE = importlib.import_module


class TestFrameEnhancer(unittest.TestCase):
    def tearDown(self):
        for module_name in TORCHVISION_MODULES:
            sys.modules.pop(module_name, None)

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

    def test_torchvision_compatibility_shim_uses_private_functional_tensor(self):
        replacement = types.ModuleType("torchvision.transforms._functional_tensor")

        def import_module(name):
            if name == "torchvision.transforms.functional_tensor":
                raise ModuleNotFoundError(
                    "No module named 'torchvision.transforms.functional_tensor'",
                    name=name,
                )
            if name == "torchvision.transforms._functional_tensor":
                return replacement
            return ORIGINAL_IMPORT_MODULE(name)

        with mock.patch("modules.enhancer.importlib.import_module", side_effect=import_module):
            with mock.patch("builtins.print") as print_mock:
                installed = install_torchvision_functional_tensor_compatibility()

        self.assertTrue(installed)
        self.assertIs(sys.modules["torchvision.transforms.functional_tensor"], replacement)
        log = "\n".join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn("Installed torchvision compatibility shim", log)

    def test_torchvision_compatibility_shim_uses_functional_fallback(self):
        replacement = types.ModuleType("torchvision.transforms.functional")

        def import_module(name):
            if name == "torchvision.transforms.functional_tensor":
                raise ModuleNotFoundError(
                    "No module named 'torchvision.transforms.functional_tensor'",
                    name=name,
                )
            if name == "torchvision.transforms._functional_tensor":
                raise ModuleNotFoundError(
                    "No module named 'torchvision.transforms._functional_tensor'",
                    name=name,
                )
            if name == "torchvision.transforms.functional":
                return replacement
            return ORIGINAL_IMPORT_MODULE(name)

        with mock.patch("modules.enhancer.importlib.import_module", side_effect=import_module):
            installed = install_torchvision_functional_tensor_compatibility()

        self.assertTrue(installed)
        self.assertIs(sys.modules["torchvision.transforms.functional_tensor"], replacement)
