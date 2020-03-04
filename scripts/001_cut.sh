#/bin/bash
ffmpeg -ss 64 -i "./data/i.mp4" -t 199 -c copy "./data/input.mp4"