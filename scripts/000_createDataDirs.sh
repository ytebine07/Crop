#/bin/bash

source ./scripts/_settings.sh

DIRS=(
    $DATA_DIR
    $IMAGE_DIR
    $CROPED_IMAGE_DIR
    $ORIGINAL_DATA_DIR
)

for DIR in ${DIRS[@]}
do
    mkdir $DIR
    echo "created -> $DIR"
done