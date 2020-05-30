import os
import ffmpeg

from decimal import Decimal, ROUND_HALF_UP


class Video:
    def __init__(self, videoPath: str):
        super().__init__()

        if not os.path.exists(videoPath):
            raise Exception("video file not found.")

        probe = ffmpeg.probe(videoPath)
        self.path: str = videoPath
        for stream in probe["streams"]:
            if stream["codec_type"] == "video":
                self.width: int = stream["width"]
                self.height: int = stream["height"]
                self.fps: Decimal = Decimal(eval(stream["r_frame_rate"])).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
