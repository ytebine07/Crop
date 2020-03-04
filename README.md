# Crop
Crop out a landscape video and make it a virtical video. Automatically crop out a person using AI.

Easy to watch on smartphone.

<img src="resources/original.gif" alt="Original" height="200">
<img src="resources/croped.gif" alt="Croped" height="200">

## Requirements
* Python 3.6.10
* Large Storage
  * Use about 23GB space per 3min freestyle.

## Preparation

### Install Python Packages
```
$ pip install -r requirementes.lock
```

### Get ImageAI Model

```
$ curl -LO https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
$ export MODEL_PATH="/path/to/resnet50_coco_best_v2.0.1.h5"
```

### Setup
Set enviroment `BASE_DIR` used by save temp data.

DO NOT SET INSIDE GIT DIR.

```
$ export BASE_DIR="/tmp"
```

Create temp data dirs
```
$ bash scripts/000_createDataDirs.sh
```

Set Original Video
* Video file requirements
  * filename  : **input.mp4**
  * FullHD(1920*1080)
  * 60fps
  * mp4

```
$ mv original.mp4 data/input.mp4
```

### Convert
#### Create images and sound file from original video.
```
$ bash scripts/002_createResource.sh
```

#### Crop images using AI :)
```
$ python main/007.py
```

#### Create croped video from croped images and original sound
```
$ bash scripts/003_finalize.sh
```

## Built With
* [ImageAI](http://imageai.org/) - Image Recognition
* [FFmpeg](https://www.ffmpeg.org/) - Create Video

## Acknowledgments
* Players, Contest Organizers and Staffs