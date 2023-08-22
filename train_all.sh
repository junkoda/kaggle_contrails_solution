#!/bin/sh

# Train final models
# About 40GB GPU RAM is required
# Takes about 40 hours per fold using A100 GPU
# Train 2 folds each

# Single-time model with 1024^2 input images
python3 src/unet1024/evaluate.py src/unet1024/unet1024.yml | exit 1
# => output/unet1024

# Four-panel model with 4 x 512^2 input images
python3 src/vit4/evaluate.py src/vit4/vit4_1024.yml
# => output/vit4_1024
