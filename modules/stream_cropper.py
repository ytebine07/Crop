import subprocess
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

    def crop(self, centers: Iterable[float]):
        import cv2

        capture = cv2.VideoCapture(self.__video.path)
        if not capture.isOpened():
            raise Exception("video file could not be opened.")

        process = self.__create_ffmpeg_process()
        try:
            for center_position in centers:
                ret, frame = capture.read()
                if not ret:
                    break

                cropped_frame = self.__crop_frame(frame, center_position)
                enhanced_frame = self.__enhancer.enhance(cropped_frame)
                resized_frame = self.__resize_frame(enhanced_frame, cv2)
                process.stdin.write(resized_frame.tobytes())
        finally:
            capture.release()
            if process.stdin:
                process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise Exception("ffmpeg failed to encode cropped video.")

        return self

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
