import os
from PIL import Image

from modules.dir import Directory
from modules.video import Video


class Cropper:
    def __init__(self, workdir: str, screen: Video):
        self.workdir: str = os.path.join(workdir, "crop")
        Directory.create(self.workdir)

        self.__video: Video = screen

    def crop(self, image_path: str, center_position):

        # 殆どのスマートフォンの画面は、9:16で構成されている。
        # 元動画から、縦横比が9:16になるように計算しています。
        # Most smartphone screens are 9:16 in size.
        # From the original video, the aspect ratio is calculated to be 9:16.
        x1 = center_position - (self.__video.height * 9 / 16 // 2)
        x2 = center_position + (self.__video.height * 9 / 16 // 2)
        file = os.path.basename(image_path)

        Image.open(image_path).crop((x1, 0, x2, self.__video.height)).save(
            os.path.join(self.workdir, file), quality=100
        )
