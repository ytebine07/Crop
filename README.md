# Crop
Crop out a landscape video and make it a virtical video. Automatically crop out a person using AI.

Easy to watch on smartphone.

<img src="resources/original.gif" alt="Original" height="200"> <img src="resources/croped.gif" alt="Croped" height="200">

- [Original] https://youtu.be/qpNQ_O5gg1Q?t=57
- [Croped]   https://youtu.be/BEyEPU3hN6s

## Requirements
* Python 3.6.10
* TensorFlow 1.15.0
* Large Storage
  * Use about 23GB space per 3min freestyle.

## Getting Started

### Setup
```
git clone https://github.com/ytebine07/Crop.git
cd Crop
./setup.sh
```

### Convert

- Replace `input.mp4` with a file name of your choice.
- The results in generated under `-w` to `final.mp4`.
```
 MSYS_NO_PATHCONV=1 \
 docker run --rm -v ${PWD}:/local \
 crop_app python crop.py \
 -f /local/input.mp4 \
 -w /local/work
```

## Usage

```
usage: crop.py [-h] -f F -w W [-a A]

[Crop] Crop out a landscape video and make it a virtical video.

optional arguments:
  -h, --help        show this help message and exit
  -f F, -file F     [required]input target video file. (default: None)
  -w W, -workdir W  [required]Directory path where script saves tmp files.
                    (default: None)
  -a A, -average A  The number of frames to be averaged over in order to make
                    the video smooth. (default: 120)
```

## Built With
* [ImageAI](https://github.com/OlafenwaMoses/ImageAI) - Image Recognition
* [FFmpeg](https://www.ffmpeg.org/) - Create Video

## Acknowledgments
* Players, Contest Organizers and Staffs