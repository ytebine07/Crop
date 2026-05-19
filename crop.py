
import argparse
import cv2
import os
import numpy as np

from tqdm import tqdm
from modules.constants import Constants as Const
from modules.video import Video
from modules.video_resource import VideoResource
from modules.actor_detector import ActorDetector, Person
from modules.convolve import Convolve
from modules.encoder import Encoder
from modules.performance_logger import PerformanceLogger
from modules.stream_cropper import StreamCropper


def main():
    parser = argparse.ArgumentParser(
        description="[Crop] Crop out a landscape video and make it a virtical video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "-file", type=argparse.FileType("r"), help="[required]input target video file.", required=True
    )
    parser.add_argument(
        "-w",
        "-workdir",
        type=str,
        help="[required]Directory path where script saves tmp files.",
        required=True
    )
    parser.add_argument(
        "-a",
        "-average",
        type=int,
        default=Const.AVERAGE_FLAMES,
        help="The number of frames to be averaged over in order to make the video smooth."
    )
    args = parser.parse_args()

    video: Video = Video(args.f.name)
    detector: ActorDetector = ActorDetector(video)
    convolve: Convolve = Convolve(args.a)
    performance_logger = PerformanceLogger(args.w)
    original_centers = []
    center_x = video.width // 2

    print("[Step. 1/4] Create Video Resources.")
    with performance_logger.step("Step. 1/4 Create Video Resources"):
        vr = VideoResource(video=video, baseDir=args.w).create_sound_only()

    print("[Step. 2/4] Detect Actor.")
    with performance_logger.step("Step. 2/4 Detect Actor"):
        capture = cv2.VideoCapture(video.path)
        try:
            if not capture.isOpened():
                raise Exception("video file could not be opened.")

            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_total = frame_count if frame_count > 0 else None
            with tqdm(total=progress_total) as progress:
                while True:
                    ret, frame = capture.read()
                    if not ret:
                        break

                    actor: Person = detector.get_actor_from_frame(frame)
                    if actor is not None:
                        original_centers.append(actor.center_x)
                        center_x = actor.center_x
                    else:
                        original_centers.append(center_x)
                    progress.update(1)
        finally:
            capture.release()

        if len(original_centers) == 0:
            raise Exception("video frame could not be read.")

        convolved_centers: list = convolve.calculate(np.array(original_centers))
    # TODO : 座標のファイル書き出し

    print("[Step. 3/4] Crop Actor.")
    with performance_logger.step("Step. 3/4 Crop Actor"):
        no_sound_path = os.path.join(args.w, Encoder.NO_SOUND_FILENAME)
        StreamCropper(video, no_sound_path).crop(tqdm(convolved_centers))

    print("[Step. 4/4] Create Croped Video.")
    with performance_logger.step("Step. 4/4 Create Croped Video"):
        Encoder(args.w, None, vr.get_sound_path(), video.fps).encode_from_video(no_sound_path)

    performance_logger.print_summary()


if __name__ == "__main__":
    main()
