import os
import subprocess
import time
from typing import Iterable, TYPE_CHECKING

from modules.constants import Constants as Const
from modules.enhancer import FrameEnhancer

if TYPE_CHECKING:
    from modules.video import Video


class StreamCropper:
    def __init__(self, video: "Video", output_path: str, enhancer=None):
        self.__video = video
        self.__output_path = output_path
        self.__crop_width = int(video.height * 9 / 16)
        self.__output_width = Const.OUTPUT_WIDTH
        self.__output_height = Const.OUTPUT_HEIGHT
        self.__enhancer = enhancer if enhancer is not None else FrameEnhancer.create_default()

    def crop(self, centers: Iterable[float], total_frames=None):
        import cv2
        from tqdm import tqdm

        capture = cv2.VideoCapture(self.__video.path)
        if not capture.isOpened():
            raise Exception("video file could not be opened.")

        progress_total = total_frames
        if progress_total is None:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_total = frame_count if frame_count > 0 else None

        process = self.__create_ffmpeg_process()
        processed_frames = 0
        started_at = time.perf_counter()
        try:
            with tqdm(
                total=progress_total,
                desc="Crop/Enhance frames",
                unit="frame",
                mininterval=1,
                leave=True,
                disable=self.__is_colab_runtime(),
            ) as progress:
                for center_position in centers:
                    ret, frame = capture.read()
                    if not ret:
                        break

                    cropped_frame = self.__crop_frame(frame, center_position)
                    enhanced_frame = self.__enhancer.enhance(cropped_frame)
                    resized_frame = self.__resize_frame(enhanced_frame, cv2)
                    resized_frame = self.__sharpen_frame(resized_frame, cv2)
                    process.stdin.write(resized_frame.tobytes())
                    processed_frames += 1
                    progress.update(1)
                    self.__print_frame_progress(
                        processed_frames,
                        progress_total,
                        started_at,
                    )
        finally:
            capture.release()
            if process.stdin:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise Exception("ffmpeg failed to encode cropped video.")

        return self

    @staticmethod
    def __is_colab_runtime():
        return (
            "COLAB_RELEASE_TAG" in os.environ
            or "COLAB_GPU" in os.environ
            or "COLAB_BACKEND_VERSION" in os.environ
        )

    def __print_frame_progress(self, processed_frames, total_frames, started_at):
        should_print = processed_frames == 1
        should_print = should_print or (
            processed_frames % Const.CROP_PROGRESS_LOG_INTERVAL_FRAMES == 0
        )
        should_print = should_print or (
            total_frames is not None and processed_frames >= total_frames
        )
        if not should_print:
            return

        elapsed = time.perf_counter() - started_at
        seconds_per_frame = elapsed / processed_frames
        frames_per_second = processed_frames / elapsed if elapsed > 0 else 0
        if total_frames is None:
            print(
                "[Progress] Crop/Enhance frames: {0} done, {1:.2f}fps".format(
                    processed_frames,
                    frames_per_second,
                ),
                flush=True,
            )
            return

        percent = processed_frames / total_frames * 100
        remaining_frames = max(total_frames - processed_frames, 0)
        eta_seconds = remaining_frames * seconds_per_frame
        print(
            "[Progress] Crop/Enhance frames: {0}/{1} ({2:.1f}%), {3:.2f}fps, eta={4}".format(
                processed_frames,
                total_frames,
                percent,
                frames_per_second,
                self.__format_seconds(eta_seconds),
            ),
            flush=True,
        )

    @staticmethod
    def __format_seconds(seconds):
        seconds = int(round(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        if hours > 0:
            return "{0:d}h{1:02d}m{2:02d}s".format(
                hours,
                minutes,
                remaining_seconds,
            )
        if minutes > 0:
            return "{0:d}m{1:02d}s".format(minutes, remaining_seconds)
        return "{0:d}s".format(remaining_seconds)

    def __crop_frame(self, frame, center_position):
        half_width = self.__crop_width // 2
        x1 = int(round(center_position)) - half_width
        x1 = max(0, min(x1, self.__video.width - self.__crop_width))
        x2 = x1 + self.__crop_width
        return frame[0:self.__video.height, x1:x2]

    def __resize_frame(self, frame, cv2):
        interpolation = cv2.INTER_AREA
        if frame.shape[1] < self.__output_width or frame.shape[0] < self.__output_height:
            interpolation = cv2.INTER_LANCZOS4
        return cv2.resize(
            frame,
            (self.__output_width, self.__output_height),
            interpolation=interpolation,
        )

    def __sharpen_frame(self, frame, cv2):
        if not Const.UNSHARP_MASK_ENABLED or Const.UNSHARP_MASK_AMOUNT <= 0:
            return frame

        blurred_frame = cv2.GaussianBlur(
            frame,
            (0, 0),
            Const.UNSHARP_MASK_SIGMA,
        )
        return cv2.addWeighted(
            frame,
            1 + Const.UNSHARP_MASK_AMOUNT,
            blurred_frame,
            -Const.UNSHARP_MASK_AMOUNT,
            0,
        )

    def __create_ffmpeg_process(self):
        return subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "{0}x{1}".format(self.__output_width, self.__output_height),
                "-r",
                str(self.__video.fps),
                "-i",
                "-",
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
                self.__output_path,
            ],
            stdin=subprocess.PIPE,
        )
