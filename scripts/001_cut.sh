#/bin/bash
ffmpeg -ss 105 -i "./data/i.mp4" -t 15 -c copy "./data/input.mp4"