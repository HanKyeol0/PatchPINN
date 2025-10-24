# scripts/load_model_exp.sh

set -e

EXPERIMENT_NAME=helmholtz2d
MODEL_NAME=ffn
EXPERIMENT_TAG=a3
DEVICE=cuda:0

FOLDER="outputs/${EXPERIMENT_NAME}_${MODEL_NAME}_${EXPERIMENT_TAG}"

TRAIN=false
EVALUATE=false
MAKE_VIDEO=true
VIDEO_FILE_NAME=low_resolution_evolution.mp4
VIDEO_GRID='{"x":60,"y":60,"t":20}'

python -m pinnlab.load_model \
  --experiment_name $EXPERIMENT_NAME \
  --model_name $MODEL_NAME \
  --folder_path $FOLDER \
  --device $DEVICE \
  --train $TRAIN \
  --evaluate $EVALUATE \
  --make_video $MAKE_VIDEO \
  --video_grid $VIDEO_GRID \
  --video_file_name $VIDEO_FILE_NAME \