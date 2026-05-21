# Crop
Crop out a landscape video and make it a virtical video. Automatically crop out a person using AI.

Easy to watch on smartphone.

<img src="resources/original.gif" alt="Original" height="200"> <img src="resources/croped.gif" alt="Croped" height="200">

- [Original] https://youtu.be/qpNQ_O5gg1Q?t=57
- [Croped]   https://youtu.be/BEyEPU3hN6s

## Requirements
* Python 3.7.6
* TensorFlow 2.4.0
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

### High quality mode on Google Colab

The command line interface is unchanged. When CUDA and Real-ESRGAN dependencies
are available, Crop automatically enhances cropped frames with GPU super
resolution before encoding `final.mp4`.

To install the Colab dependencies:

```
pip install -r requirements-colab.txt
```

Colab must use a GPU runtime. Verify PyTorch can see CUDA before running Crop:

```
!nvidia-smi
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

If `torch.cuda.is_available()` prints `False`, switch Colab to a GPU runtime
and restart the runtime before installing dependencies again.

To disable GPU enhancement and use standard high-quality resizing:

```
CROP_ENHANCER=off python crop.py -f input.mp4 -w /content/work
```

To use the video super-resolution path with RealBasicVSR, prepare the official
RealBasicVSR repository in Colab and run Crop with `CROP_ENHANCER=realbasicvsr`:

```
cd /content/Crop
chmod 755 ./setup_colab_realbasicvsr.sh
./setup_colab_realbasicvsr.sh

CROP_ENHANCER=realbasicvsr python crop.py -f input.mp4 -w /content/work
```

If your RealBasicVSR checkout or checkpoint is in a different location, set:

```
REAL_BASIC_VSR_REPO=/content/RealBasicVSR
REAL_BASIC_VSR_CONFIG=configs/realbasicvsr_x4.py
REAL_BASIC_VSR_CHECKPOINT=/content/RealBasicVSR/checkpoints/RealBasicVSR_x4.pth
REAL_BASIC_VSR_MAX_SEQ_LEN=30
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

## For Develop
When developing with `Visual Studio Code` + `Remote Containers`

1. Open command pallet
2. Chose `Remote-containers: Open Folder in Cotainer...`, start building the environment.

### tips
When you start `crop.py`, `-w` directory should be outside git.  
e.g. `/tmp/workdir`  

Because it`s slower inside the git.



## Built With
* [ImageAI](https://github.com/OlafenwaMoses/ImageAI) - Image Recognition
* [FFmpeg](https://www.ffmpeg.org/) - Create Video

## Acknowledgments
* Players, Contest Organizers and Staffs
