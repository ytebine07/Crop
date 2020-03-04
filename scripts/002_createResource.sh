#/bin/bash

source ./scripts/_settings.sh

# create sound.mp4
ffmpeg -i  $ORIGINAL_PATH -vn -acodec copy -map 0:1 "$ORIGINAL_DATA_DIR/sound.mp4"

# create images
ffmpeg -i $ORIGINAL_PATH -vcodec png "$IMAGE_DIR/image_%5d.png"
