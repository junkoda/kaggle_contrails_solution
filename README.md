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

y_pred = w1 * y_vit4 + w2 * y_unet1024 > 0.5

See `notebook/` for the weights.


### unet5

The base single-time U-Net model with 256 or 512 input image.

### unet1024

U-Net model with 1024x1024 input image and 512x512 segmentation target.
Almost same as the base model, but one less upscale decoder layer.

### vit4

The input is four panels of 512x512 inputs at t=1,2,3,4, concatenated to one 1024x1024 input image. Target is 512x512 image for t=4.

## Weights

The inference code and model weights are online on Kaggle:

- https://www.kaggle.com/code/junkoda/contrails-submit
- https://www.kaggle.com/datasets/junkoda/contrails-src
- https://www.kaggle.com/datasets/junkoda/contrails-weights


## Training

- entry_points.md: Instructions how to train the model
- SETTINGS.json: Directory configuration for training
- Training the final model would take 40 hours per fold with A100 GPU (40GB RAM)


## Software used

- Python 3.10.10
- CUDA 11.7 (NVIDIA driver Version: 530.30.02)
- PyTorch 2.0.1
- timm 0.9.2
- segmentation-models-pytorch 0.3.3
- albumentations 1.3.0
- h5py 3.9.0
- Standard packages: numpy, pandas, tqdm
- requirements.txt: List of all python modules, more than sufficient


## LICENSE

MIT
