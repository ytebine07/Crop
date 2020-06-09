#/bin/bash
MODEL_DIR='./model'
MODEL_FILE='resnet50_coco_best_v2.0.1.h5'
DOWNLOAD_URL='https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5'
MODEL_FILE_PATH=$MODEL_DIR/$MODEL_FILE

echo "[Crop]Setup Script."
echo "- Download ImageAI model."
if [ ! -e  $MODEL_FILE_PATH ];then
    echo "  - Downloading....."
    curl -LO $DOWNLOAD_URL
    mv $MODEL_FILE $MODEL_DIR/
else
    echo "  - Already Exists. Skip Download."
fi

docker-compose up --build -d