import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from modules.performance_logger import PerformanceLogger


class TestPerformanceLogger(unittest.TestCase):
    def test_step_outputs_elapsed_size_and_delta(self):
        with tempfile.TemporaryDirectory() as workdir:
            logger = PerformanceLogger(workdir)
            output = io.StringIO()

            with redirect_stdout(output):
                with logger.step("Step Test"):
                    with open(os.path.join(workdir, "result.bin"), "wb") as f:
                        f.write(b"0" * 2048)

            logs = output.getvalue()
            self.assertIn("[Metrics] Step Test:", logs)
            self.assertIn("elapsed=", logs)
            self.assertIn("workdir_size=2.00KB", logs)
            self.assertIn("delta=2.00KB", logs)

    def test_summary_outputs_directory_breakdown(self):
        with tempfile.TemporaryDirectory() as workdir:
            child_dir = os.path.join(workdir, "child")
            os.mkdir(child_dir)
            with open(os.path.join(child_dir, "result.bin"), "wb") as f:
                f.write(b"0" * 1024)

            logger = PerformanceLogger(workdir)
            output = io.StringIO()

            with redirect_stdout(output):
                with logger.step("Step Test"):
                    pass
                logger.print_summary()

            logs = output.getvalue()
            self.assertIn("[Metrics] Summary", logs)
            self.assertIn("[Metrics] Directory size breakdown", logs)
            self.assertIn(workdir, logs)
            self.assertIn(child_dir, logs)
