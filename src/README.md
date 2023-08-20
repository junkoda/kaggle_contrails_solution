src
========

Source files.

* script/prepare.py
  - Preprocess training data

* unet5
  - Base U-Net model. Not in final model but contains common files.
  - input is single-time 256 or 512 pixel image per dimension

* unet1024
  - Single-time U-Net model with 1024x1024 input

* vit4
  - 4-panel U-Net model using t=1,2,3,4

* submit
  - inference code https://www.kaggle.com/datasets/junkoda/contrails-src
