# Kaggle Contrails 1st place solution

This is the 1st-place solution for:

Google Research - Identify Contrails to Reduce Global Warming

https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming

Solution post:

https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430618


## Models

The final model is the ensemble of unet1024 model and vit4 model.

- Version 43: 2 fold + 2 fold is the winning solution (Private LB 0.72432)
- Version 47: 5 fold + 5 fold only differ by random fluctuation (Private 0.72420)

y = w1 * y_vit4 + w2 * y_unet1024 > 0.5

See `notebook/` for the weights.


### unet5

The base single-time U-Net model with 256 or 512 input image.

### unet1024

U-Net model with 1024x1024 input image and 512x512 segmentation target.
Almost same as the base model, but one less upscale decoder layer.

### vit4

The input is four panels of 512x512 inputs at t=1,2,3,4, concatenated to one 1024x1024 input image. Target is 512x512 image for t=4.

## Weights

Available in Kaggle datasets.

https://www.kaggle.com/datasets/junkoda/contrails-weights


## Training

- entry_points.md: Instructions how to train the model
- SETTINGS.json: Directory configuration for training
- requirements.txt: All python modules, more than sufficient

## LICENCE

MIT
