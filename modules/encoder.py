import os
import ffmpeg
from typing import List


class Encoder:

    NO_SOUND_FILENAME = "nosound.mp4"
    FINAL_FILENAME = "final.mp4"

    def __init__(self, baseDir: str, images: List[str], sound: str, fps):
        self.__work_dir = baseDir
        self.__images_path = images
        self.__sound_path = sound
        self.__fps = fps

        self.__no_sound_filepath = os.path.join(self.__work_dir, self.NO_SOUND_FILENAME)
        self.__final_filepath = os.path.join(self.__work_dir, self.FINAL_FILENAME)

    def encode(self):
        self.__encode_no_sound_video()
        self.__encode_final_video()
        return self

    def __encode_no_sound_video(self):
        (
            ffmpeg.input(self.__images_path, r=self.__fps)
            .output(
                r=self.__fps,
                vcodec="libx264",
                pix_fmt="yuv420p",
                filename=self.__no_sound_filepath,
            )
            .run(overwrite_output=True)
        )

    def __encode_final_video(self):
        video_stream = ffmpeg.input(self.__no_sound_filepath)
        sound_stream = ffmpeg.input(self.__sound_path)
        ffmpeg.output(
            video_stream,
            sound_stream,
            vcodec="copy",
            acodec="copy",
            filename=self.__final_filepath,
        ).run(overwrite_output=True)
