#/bin/bash
ffmpeg -ss 40 -i "./data/i.mp4" -t 15 -c copy "./data/input.mp4"