
import argparse
import numpy as np

from tqdm import tqdm
from modules.constants import Constants as Const
from modules.video import Video
from modules.video_resource import VideoResource
from modules.actor_detector import ActorDetector, Person
from modules.convolve import Convolve
from modules.cropper import Cropper
from modules.encoder import Encoder


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
    original_centers = []
    center_x = video.width // 2

    print("[Step. 1/4] Create Video Resources.")
    vr = VideoResource(video=video, baseDir=args.w).create()

    print("[Step. 2/4] Detect Actor.")
    for image_path in tqdm(vr.get_image_paths()):
        actor: Person = detector.get_actor(image_path)
        if actor is not None:
            original_centers.append(actor.center_x)
            center_x = actor.center_x
        else:
            original_centers.append(center_x)

    convolved_centers: list = convolve.calculate(np.array(original_centers))
    zzz = list(zip(vr.get_image_paths(), original_centers, convolved_centers))
    # TODO : 座標のファイル書き出し

    print("[Step. 3/4] Crop Actor.")
    cropper = Cropper(args.w, video)
    for image_path, _, center_position in tqdm(zzz):
        cropper.crop(image_path, center_position)

    print("[Step. 4/4] Create Croped Video.")
    Encoder(args.w, cropper.get_images_path(), vr.get_sound_path(), video.fps).encode()


if __name__ == "__main__":
    main()
