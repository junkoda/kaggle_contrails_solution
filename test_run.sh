#!/bin/sh

set -x

# Prepare
if [ ! -d data/compact4 ]; then
  python3 src/script/prepare.py train       || exit 1
  python3 src/script/prepare.py validation  || exit 1
fi

# Training debug run

python3 src/unet1024/evaluate.py src/unet1024/devel.yml --debug || exit 1

python3 src/vit4/evaluate.py src/vit4/devel.yml --debug || exit 1

echo "Test run OK"

# Debug run outputs to output/debug/

# Train small model (256) for practice
# python3 src/unet5/evaluate.py src/unet5/resnest26.yml
# This takes 2.4 minutes per epoch, 2 hours for 50 epochs
# using RTX3090 GPU and M2 SSD (3500 MB/s)

# Larger model
# python3 src/unet5/evaluate.py src/unet5/maxvit_tiny.yml --debug
