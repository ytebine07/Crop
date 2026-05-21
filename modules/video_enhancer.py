import glob
import os
import subprocess

from modules.constants import Constants as Const
from modules.dir import Directory


class NoOpVideoEnhancer:
    name = "none"

    def enhance(self, input_video_path: str, output_video_path: str):
        if input_video_path != output_video_path:
            self.__copy_video(input_video_path, output_video_path)
        return output_video_path

    @staticmethod
    def __copy_video(input_video_path: str, output_video_path: str):
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_video_path, "-c", "copy", output_video_path],
            check=True,
        )


class RealBasicVSREnhancer:
    name = "realbasicvsr"

    def __init__(
        self,
        workdir: str,
        fps,
        repo_dir=None,
        config_path=None,
        checkpoint_path=None,
        max_seq_len=None,
    ):
        self.__workdir = workdir
        self.__fps = fps
        self.__repo_dir = repo_dir or os.environ.get(
            Const.REAL_BASIC_VSR_REPO_ENV,
            Const.REAL_BASIC_VSR_REPO_DIR,
        )
        self.__config_path = config_path or os.environ.get(
            Const.REAL_BASIC_VSR_CONFIG_ENV,
            Const.REAL_BASIC_VSR_CONFIG,
        )
        self.__checkpoint_path = checkpoint_path or os.environ.get(
            Const.REAL_BASIC_VSR_CHECKPOINT_ENV,
            Const.REAL_BASIC_VSR_CHECKPOINT,
        )
        self.__max_seq_len = int(
            max_seq_len
            or os.environ.get(
                Const.REAL_BASIC_VSR_MAX_SEQ_LEN_ENV,
                Const.REAL_BASIC_VSR_MAX_SEQ_LEN,
            )
        )
        self.__input_frames_dir = os.path.join(workdir, "realbasicvsr_input")
        self.__output_frames_dir = os.path.join(workdir, "realbasicvsr_output")

    def enhance(self, input_video_path: str, output_video_path: str):
        self.__validate()
        Directory.create(self.__input_frames_dir)
        Directory.create(self.__output_frames_dir)

        self.__extract_frames(input_video_path)
        self.__run_inference()
        self.__encode_video(output_video_path)
        return output_video_path

    def __validate(self):
        script_path = self.__script_path()
        config_path = self.__resolve_repo_path(self.__config_path)
        missing_paths = []
        for path in (script_path, config_path, self.__checkpoint_path):
            if not os.path.exists(path):
                missing_paths.append(path)

        if len(missing_paths) > 0:
            raise RuntimeError(
                "RealBasicVSR is not ready. Missing: {0}".format(
                    ", ".join(missing_paths)
                )
            )

    def __extract_frames(self, input_video_path: str):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_video_path,
                self.__input_frame_pattern(),
            ],
            check=True,
        )

    def __run_inference(self):
        subprocess.run(
            [
                "python",
                self.__script_path(),
                self.__resolve_repo_path(self.__config_path),
                self.__checkpoint_path,
                self.__input_frames_dir,
                self.__output_frames_dir,
                "--max-seq-len={0}".format(self.__max_seq_len),
                "--is_save_as_png",
                "--fps={0}".format(self.__fps),
            ],
            check=True,
            cwd=self.__repo_dir,
        )

    def __encode_video(self, output_video_path: str):
        if len(glob.glob(os.path.join(self.__output_frames_dir, "*.png"))) == 0:
            raise RuntimeError("RealBasicVSR did not create output frames.")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(self.__fps),
                "-i",
                self.__output_frame_pattern(),
                "-vf",
                "scale={0}:{1}".format(Const.OUTPUT_WIDTH, Const.OUTPUT_HEIGHT),
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                Const.VIDEO_PRESET,
                "-crf",
                str(Const.VIDEO_CRF),
                "-profile:v",
                Const.VIDEO_PROFILE,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_video_path,
            ],
            check=True,
        )

    def __script_path(self):
        return os.path.join(self.__repo_dir, "inference_realbasicvsr.py")

    def __resolve_repo_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.__repo_dir, path)

    def __input_frame_pattern(self):
        return os.path.join(self.__input_frames_dir, "%08d.png")

    def __output_frame_pattern(self):
        return os.path.join(self.__output_frames_dir, "%08d.png")


class VideoEnhancer:
    @staticmethod
    def is_requested():
        return (
            os.environ.get(Const.ENHANCER_ENV, Const.ENHANCER_AUTO).lower()
            == Const.ENHANCER_REAL_BASIC_VSR
        )

    @staticmethod
    def create_default(workdir: str, fps):
        mode = os.environ.get(Const.ENHANCER_ENV, Const.ENHANCER_AUTO).lower()
        if mode == Const.ENHANCER_REAL_BASIC_VSR:
            print("[Enhancer] Using realbasicvsr video enhancer.")
            return RealBasicVSREnhancer(workdir, fps)
        return NoOpVideoEnhancer()
