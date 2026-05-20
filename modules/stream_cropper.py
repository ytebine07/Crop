import subprocess
from typing import Iterable

import cv2

from modules.video import Video


class StreamCropper:
    def __init__(self, video: Video, output_path: str):
        self.__video = video
        self.__output_path = output_path
        self.__crop_width = int(video.height * 9 / 16)
        self.__output_width = 720
        self.__output_height = 1280

    def crop(self, centers: Iterable[float]):
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
                resized_frame = cv2.resize(
                    cropped_frame,
                    (self.__output_width, self.__output_height),
                    interpolation=cv2.INTER_AREA,
                )
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
                "-pix_fmt",
                "yuv420p",
                self.__output_path,
            ],
            stdin=subprocess.PIPE,
        )
