#!/bin/sh

# Prepare
if [ ! -d data/compact4 ]; then
  python3 src/script/prepare.py train
  python3 src/script/prepare.py validation
fi

# Training debug run
# python3 src/unet5/evaluate.py src/unet5/maxvit_tiny.yml --debug

python3 src/unet1024/evaluate.py src/unet1024/devel.yml --debug || exit 1

python3 src/vit4/evaluate.py src/vit4/devel.yml --debug

# Debug run outputs to output/debug/

# Train small model
# python3 src/unet5/evaluate.py src/unet5/resnest.yml
# This takes *** with RTX3090 GPU and M2 SSD (3500 MB/s)
# score ***