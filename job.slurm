#!/bin/bash
#SBATCH --job-name=2DSUGAR
#SBATCH --partition=dgx
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --cpus-per-task=20
#SBATCH --gres=shard:30

source activate base
conda activate colmap

RESOLUTION=1 FPS=6 ./train_video.sh datasets/VID_20261020_204406.mp4 kolonka
