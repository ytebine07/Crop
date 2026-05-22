import glob
import os
import shutil
import subprocess
import time

from modules.constants import Constants as Const
from modules.dir import Directory


class NoOpVideoEnhancer:
    name = "none"

    def validate(self):
        return self

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
        progress_interval_seconds=None,
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
        self.__max_seq_len = max(
            1,
            int(
                max_seq_len
                or os.environ.get(
                    Const.REAL_BASIC_VSR_MAX_SEQ_LEN_ENV,
                    Const.REAL_BASIC_VSR_MAX_SEQ_LEN,
                )
            ),
        )
        self.__progress_interval_seconds = float(
            progress_interval_seconds
            if progress_interval_seconds is not None
            else Const.REAL_BASIC_VSR_PROGRESS_INTERVAL_SECONDS
        )
        self.__input_frames_dir = os.path.join(workdir, "realbasicvsr_input")
        self.__output_frames_dir = os.path.join(workdir, "realbasicvsr_output")
        self.__chunks_dir = os.path.join(workdir, "realbasicvsr_chunks")

    def validate(self):
        script_path = self.__script_path()
        config_path = self.__resolve_repo_path(self.__config_path)
        missing_paths = []
        for path in (script_path, config_path, self.__checkpoint_path):
            if not os.path.exists(path):
                missing_paths.append(path)

        if len(missing_paths) > 0:
            raise RuntimeError(
                "RealBasicVSR is not ready. Missing: {0}. "
                "Prepare the official RealBasicVSR repo and checkpoint, or set "
                "{1}, {2}, and {3}.".format(
                    ", ".join(missing_paths),
                    Const.REAL_BASIC_VSR_REPO_ENV,
                    Const.REAL_BASIC_VSR_CONFIG_ENV,
                    Const.REAL_BASIC_VSR_CHECKPOINT_ENV,
                )
            )
        return self

    def enhance(self, input_video_path: str, output_video_path: str):
        self.validate()
        self.__reset_dir(self.__input_frames_dir)
        self.__reset_dir(self.__output_frames_dir)
        self.__reset_dir(self.__chunks_dir)

        print("[Progress] RealBasicVSR: start video enhancement.", flush=True)
        self.__extract_frames(input_video_path)
        self.__run_inference()
        self.__encode_video(output_video_path)
        print("[Progress] RealBasicVSR: finished video enhancement.", flush=True)
        return output_video_path

    @staticmethod
    def __reset_dir(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        Directory.create(directory)

    def __extract_frames(self, input_video_path: str):
        print("[Progress] RealBasicVSR extract frames: start.", flush=True)
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
        print(
            "[Progress] RealBasicVSR extract frames: {0} frames ready.".format(
                self.__count_pngs(self.__input_frames_dir)
            ),
            flush=True,
        )

    def __run_inference(self):
        env = self.__subprocess_env()
        input_frame_paths = self.__input_frame_paths()
        total_frames = len(input_frame_paths)
        print(
            "[Progress] RealBasicVSR inference: start. total_frames={0}, max_seq_len={1}".format(
                total_frames,
                self.__max_seq_len,
            ),
            flush=True,
        )
        started_at = time.monotonic()

        for chunk_index, chunk_frame_paths in enumerate(
            self.__chunks(input_frame_paths, self.__max_seq_len),
            start=1,
        ):
            self.__run_inference_chunk(
                chunk_index,
                chunk_frame_paths,
                env,
                total_frames,
                started_at,
            )

        print(
            "[Progress] RealBasicVSR inference: finished. output_frames={0}".format(
                self.__count_pngs(self.__output_frames_dir)
            ),
            flush=True,
        )

    def __run_inference_chunk(
        self,
        chunk_index,
        chunk_frame_paths,
        env,
        total_frames,
        started_at,
    ):
        chunk_dir = self.__prepare_chunk_dir(chunk_index, chunk_frame_paths)
        print(
            "[Progress] RealBasicVSR inference chunk: {0}, frames={1}".format(
                chunk_index,
                len(chunk_frame_paths),
            ),
            flush=True,
        )
        command = self.__inference_command(chunk_dir)
        try:
            self.__run_inference_process(command, env, total_frames, started_at)
        except subprocess.CalledProcessError:
            if len(chunk_frame_paths) <= 1:
                raise
            self.__remove_output_frames(chunk_frame_paths)
            midpoint = len(chunk_frame_paths) // 2
            print(
                "[Progress] RealBasicVSR inference chunk: {0} failed; retry with {1}+{2} frames.".format(
                    chunk_index,
                    midpoint,
                    len(chunk_frame_paths) - midpoint,
                ),
                flush=True,
            )
            self.__run_inference_chunk(
                chunk_index,
                chunk_frame_paths[:midpoint],
                env,
                total_frames,
                started_at,
            )
            self.__run_inference_chunk(
                chunk_index,
                chunk_frame_paths[midpoint:],
                env,
                total_frames,
                started_at,
            )

    def __run_inference_process(self, command, env, total_frames, started_at):
        process = subprocess.Popen(
            command,
            cwd=self.__repo_dir,
            env=env,
        )
        last_logged_at = started_at - self.__progress_interval_seconds
        last_logged_count = -1

        while True:
            return_code = process.poll()
            now = time.monotonic()
            completed_frames = self.__count_pngs(self.__output_frames_dir)
            should_log = (
                completed_frames != last_logged_count
                or now - last_logged_at >= self.__progress_interval_seconds
                or return_code is not None
            )
            if should_log:
                self.__print_inference_progress(
                    completed_frames,
                    total_frames,
                    now - started_at,
                )
                last_logged_at = now
                last_logged_count = completed_frames
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                break
            time.sleep(max(self.__progress_interval_seconds, 1))

    def __inference_command(self, input_frames_dir):
        return [
            "python",
            self.__script_path(),
            self.__inference_config_path(),
            self.__checkpoint_path,
            input_frames_dir,
            self.__output_frames_dir,
            "--max_seq_len={0}".format(self.__max_seq_len),
            "--is_save_as_png=True",
            "--fps={0}".format(self.__fps),
        ]

    def __prepare_chunk_dir(self, chunk_index, frame_paths):
        chunk_dir = os.path.join(
            self.__chunks_dir,
            "{0:06d}".format(chunk_index),
        )
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        Directory.create(chunk_dir)
        for frame_path in frame_paths:
            link_path = os.path.join(chunk_dir, os.path.basename(frame_path))
            try:
                os.symlink(frame_path, link_path)
            except OSError:
                shutil.copy2(frame_path, link_path)
        return chunk_dir

    @staticmethod
    def __chunks(items, size):
        for index in range(0, len(items), size):
            yield items[index : index + size]

    def __input_frame_paths(self):
        return sorted(glob.glob(os.path.join(self.__input_frames_dir, "*.png")))

    def __remove_output_frames(self, input_frame_paths):
        for input_frame_path in input_frame_paths:
            output_frame_path = os.path.join(
                self.__output_frames_dir,
                os.path.basename(input_frame_path),
            )
            if os.path.exists(output_frame_path):
                os.remove(output_frame_path)

    def __print_inference_progress(self, completed_frames, total_frames, elapsed_seconds):
        fps = completed_frames / elapsed_seconds if elapsed_seconds > 0 else 0
        if total_frames > 0:
            percent = completed_frames / total_frames * 100
            remaining_frames = max(total_frames - completed_frames, 0)
            eta = self.__format_seconds(remaining_frames / fps if fps > 0 else None)
            print(
                "[Progress] RealBasicVSR inference: {0}/{1} ({2:.1f}%), {3:.2f}fps, eta={4}".format(
                    completed_frames,
                    total_frames,
                    percent,
                    fps,
                    eta,
                ),
                flush=True,
            )
            return

        print(
            "[Progress] RealBasicVSR inference: {0} frames written, elapsed={1}".format(
                completed_frames,
                self.__format_seconds(elapsed_seconds),
            ),
            flush=True,
        )

    @staticmethod
    def __format_seconds(seconds):
        if seconds is None:
            return "unknown"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        if hours > 0:
            return "{0}h{1:02d}m{2:02d}s".format(hours, minutes, remaining_seconds)
        if minutes > 0:
            return "{0}m{1:02d}s".format(minutes, remaining_seconds)
        return "{0}s".format(remaining_seconds)

    @staticmethod
    def __count_pngs(directory):
        return len(glob.glob(os.path.join(directory, "*.png")))

    def __subprocess_env(self):
        env = os.environ.copy()
        compat_dir = os.path.join(self.__workdir, "realbasicvsr_py_compat")
        Directory.create(compat_dir)
        sitecustomize_path = os.path.join(compat_dir, "sitecustomize.py")
        with open(sitecustomize_path, "w") as sitecustomize:
            sitecustomize.write(
                "\n".join(
                    [
                        "import pkgutil, zipimport",
                        "if not hasattr(pkgutil, 'ImpImporter'):",
                        "    pkgutil.ImpImporter = zipimport.zipimporter",
                        "if not hasattr(pkgutil, 'ImpLoader'):",
                        "    pkgutil.ImpLoader = zipimport.zipimporter",
                        "try:",
                        "    import numpy as _np",
                        "    for _name, _value in {",
                        "        'bool': bool,",
                        "        'bool8': _np.bool_,",
                        "        'int': int,",
                        "        'int0': _np.intp,",
                        "        'float': float,",
                        "        'float_': _np.float64,",
                        "        'complex': complex,",
                        "        'complex_': _np.complex128,",
                        "        'object': object,",
                        "        'uint0': _np.uintp,",
                        "    }.items():",
                        "        if _name not in _np.__dict__:",
                        "            setattr(_np, _name, _value)",
                        "except Exception:",
                        "    pass",
                        "",
                    ]
                )
            )
        env["PYTHONPATH"] = self.__prepend_path(compat_dir, env.get("PYTHONPATH", ""))
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        return env

    def __inference_config_path(self):
        source_config_path = self.__resolve_repo_path(self.__config_path)
        config_path = os.path.join(self.__workdir, "realbasicvsr_inference_config.py")
        with open(source_config_path) as source_config:
            config_content = source_config.read()
        with open(config_path, "w") as inference_config:
            inference_config.write(config_content)
            inference_config.write(
                "\n".join(
                    [
                        "",
                        "# Crop inference override: avoid constructing training-only modules.",
                        "model['discriminator'] = None",
                        "model['gan_loss'] = None",
                        "model['pixel_loss'] = None",
                        "model['cleaning_loss'] = None",
                        "model['perceptual_loss'] = None",
                        "",
                    ]
                )
            )
        return config_path

    @staticmethod
    def __prepend_path(path, current):
        if current:
            return path + os.pathsep + current
        return path

    def __encode_video(self, output_video_path: str):
        output_frame_count = self.__count_pngs(self.__output_frames_dir)
        if output_frame_count == 0:
            raise RuntimeError("RealBasicVSR did not create output frames.")

        print(
            "[Progress] RealBasicVSR encode video: start. frames={0}".format(
                output_frame_count
            ),
            flush=True,
        )
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
        print("[Progress] RealBasicVSR encode video: finished.", flush=True)

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
