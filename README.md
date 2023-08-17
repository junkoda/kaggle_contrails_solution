# Kaggle Contrails 1st place solution
This is the 1st-place solution for:

Google Research - Identify Contrails to Reduce Global Warming

https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming

Solution post:

https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430618


## Models

The final model is the ensemble of unet1024 model and vit4 model.

- Ensemble 1: 2 fold + 2 fold is the winning solution (Private LB 0.72432)
- Ensemble 2: 5 fold + 5 fold only differ by random fluctuation (Private 0.72420)

y = w1 * y_vit4 + w2 * y_unet1024 > 0.5

### Ensemble 1

Version 43

```text
2 fold + 2 fold mean

Local CV:   0.70639
Public LB:  0.72531
Private LB: 0.72432

w1: 0.6465366913319027
w2: 0.4531613974630522
```

### Ensemble 2

Version 47

```text
5 fold + 5 fold mean

Local CV:   0.70601
Public LB:  0.72544
Private LB: 0.72420

w1: 0.6669411076255138
w2: 0.42019871465275055
```

### unet5

The base single-time U-Net model for 256 or 512 input image.

### unet1024

U-Net model with 1024x1024 input image and 512x512 segmentation target.
Almost same as the base model, but one less upscale decoder layer.

### vit4

The input is four panels of 512x512 inputs at t=1,2,3,4, concatenated to one 1024x1024 input image. Target is 512x512 image for t=4.

## Weights

Available in Kaggle datasets.

https://www.kaggle.com/datasets/junkoda/contrails-weights


## Winning solution

Files required for the winning solution

- SETTINGS.json: Directory configuration
- entry_points.md: Instructions how to use or train the model
- requirements.txt: All modules installed, more than sufficient
- directory_structure: List of directories

## LICENCE

MIT
