import os
import glob
import ffmpeg
from typing import List
from modules.video import Video
from modules.dir import Directory


class VideoResource:
    def __init__(self, video: Video, baseDir: str):
        self.workDir = os.path.join(baseDir, "original")
        Directory.create(self.workDir)

        self.__video = video
        self.__stream = ffmpeg.input(video.path)

    def create(self):
        self.__extract_images()
        self.__extract_sound()
        return self

    def get_image_paths(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self.workDir, "*png")))

    def get_image_path(self) -> str:
        return os.path.join(self.workDir, "*png")

    def get_sound_path(self) -> str:
        return os.path.join(self.workDir, "sound.mp4")

    def __extract_images(self):
        ffmpeg.output(self.__stream, os.path.join(self.workDir, "image_%5d.png")).run()

    def __extract_sound(self):
        (
            ffmpeg.input(self.__video.path)
            .output(
                acodec="copy",
                map="0:1",
                filename=os.path.join(self.workDir, "sound.mp4"),
            )
            .run(capture_stdout=True)
        )
