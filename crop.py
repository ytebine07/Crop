
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
from modules.enhancer import NoOpEnhancer
from modules.performance_logger import PerformanceLogger
from modules.stream_cropper import StreamCropper
from modules.video_enhancer import VideoEnhancer


def main():
    parser = argparse.ArgumentParser(
        description="[Crop] Crop out a landscape video and make it a virtical video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f", "-file", type=str, help="[required]input target video file.", required=True
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
    parser.add_argument(
        "-i",
        "-interval",
        type=int,
        default=Const.DETECT_INTERVAL_FRAMES,
        help="The frame interval for actor detection. Intermediate frames are interpolated."
    )
    args = parser.parse_args()
    if args.i < 1:
        raise Exception("detect interval must be greater than or equal to 1.")

    video: Video = Video(args.f)
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
        detected_frame_indexes = []
        detected_centers = []
        frame_index = 0
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

                    if frame_index % args.i == 0:
                        actor: Person = detector.get_actor_from_frame(frame)
                        if actor is not None:
                            center_x = actor.center_x
                        detected_frame_indexes.append(frame_index)
                        detected_centers.append(center_x)

                    frame_index += 1
                    progress.update(1)
        finally:
            capture.release()

        if frame_index == 0:
            raise Exception("video frame could not be read.")

        if len(detected_centers) == 0:
            detected_frame_indexes.append(0)
            detected_centers.append(center_x)

        original_centers = np.interp(
            np.arange(frame_index),
            np.array(detected_frame_indexes),
            np.array(detected_centers),
        )
        convolved_centers: list = convolve.calculate(np.array(original_centers))
    # TODO : 座標のファイル書き出し

    print("[Step. 3/4] Crop Actor.")
    with performance_logger.step("Step. 3/4 Crop Actor"):
        no_sound_path = os.path.join(args.w, Encoder.NO_SOUND_FILENAME)
        if VideoEnhancer.is_requested():
            video_enhancer = VideoEnhancer.create_default(args.w, video.fps)
            video_enhancer.validate()
            cropped_raw_path = os.path.join(args.w, Const.CROPPED_RAW_FILENAME)
            raw_crop_width = int(video.height * 9 / 16)
            if raw_crop_width % 2 != 0:
                raw_crop_width += 1
            StreamCropper(
                video,
                cropped_raw_path,
                enhancer=NoOpEnhancer(),
                output_width=raw_crop_width,
                output_height=video.height,
                sharpen=False,
            ).crop(
                convolved_centers,
                total_frames=len(convolved_centers),
            )
            video_enhancer.enhance(
                cropped_raw_path,
                no_sound_path,
            )
        else:
            StreamCropper(video, no_sound_path).crop(
                convolved_centers,
                total_frames=len(convolved_centers),
            )

    print("[Step. 4/4] Create Croped Video.")
    with performance_logger.step("Step. 4/4 Create Croped Video"):
        Encoder(args.w, None, vr.get_sound_path(), video.fps).encode_from_video(no_sound_path)

    performance_logger.print_summary()


if __name__ == "__main__":
    main()
