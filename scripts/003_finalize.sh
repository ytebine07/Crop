#/bin/bash

source ./scripts/_settings.sh

# create croped mp4
ffmpeg -r 60 -i "$CROPED_IMAGE_DIR/image_%05d.png" -vcodec libx264 -pix_fmt yuv420p -r 60 "$DATA_DIR/croped_nosound.mp4"

# merge sound
ffmpeg -i "$DATA_DIR/croped_nosound.mp4" -i "$ORIGINAL_DATA_DIR/sound.mp4" -c:v copy -map 0:v:0 -map 1:a:0 "./data/final.mp4"

