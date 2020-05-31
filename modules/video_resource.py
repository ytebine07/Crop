import os
import glob
import ffmpeg
from modules.video import Video


class VideoResource:
    def __init__(self, video: Video, path: str):
        self.__video = video
        self.__path = path
        self.__stream = ffmpeg.input(video.path)

    def create(self):
        self.__stream = ffmpeg.output(
            self.__stream, os.path.join(self.__path, "image_%5d.png")
        )
        ffmpeg.run(self.__stream)
        return self

    def get_video_resources(self) -> list:
        return sorted(glob.glob(os.path.join(self.__path, "*png")))
