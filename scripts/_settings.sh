#/bin/bash


if [ ! -v BASE_DIR ]; then
    echo ""
    echo "[ERROR]undefined env BASE_DIR"
    echo "-------------------------"
    echo "export BASE_DIR=/tmp"
    echo "-------------------------"
    echo ""
    exit 1
fi

DATA_DIR=$BASE_DIR"/set"
IMAGE_DIR=$DATA_DIR"/images"
CROPED_IMAGE_DIR=$DATA_DIR"/croped"
ORIGINAL_DATA_DIR=$DATA_DIR"/origin"

ORIGINAL_PATH="./data/input.mp4"