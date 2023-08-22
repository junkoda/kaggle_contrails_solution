Entry Points
============

This note explains how to train models. See also `winning_solution.md`.

# Assumptions

- `SETTINGS.json` exists in this directory.
  * different settings file can be specified with `--settings=filename` for python scripts.

- `INPUT_DIR`, `DATA_DIR`, and `OUTPUT_DIR` specified in `SETTINGS.json` exists
  * the codes do not create these directories automatically

- Kaggle data are in `INPUT_DIR/google-research-identify-contrails-reduce-global-warming`
  * Put the data (or symbolic link to the data) in `./input` directory,
  * or, set `INPUT_DIR` in `SETTINGS.json`
- About 70 GB of free disk space for `DATA_DIR`
- About 128MB per model weight, 512MB for 2 folds + 2 folds


# 1. Prepare

The training and validation data needs to be converted for efficient data loading.

```bash
$ python3 src/script/convert_data_compact4.py train
$ python3 src/script/convert_data_compact4.py validation
```

- Read data from `INPUT_DIR/google-research-identify-contrails-reduce-global-warming`
- Output HDF5 to `DATA_DIR/compact4`


# 2. Train

Test run:

```
$ sh test_run.sh
```

The run is checked with GPU RTX3090 (24GB RAM); 16GB is insufficient.

## Full training

Training the final models requires about 40GB RAM (24 GB is insufficient).
The required RAM can be reduced with smaller batch size, but the model performance could be different.

```bash
$ python3 src/unet1024/evaluate.py src/unet1024/unet1024.yml 
$ python3 src/vit4/evaluate.py src/vit4/vit4_1024.yml
```

- Reads data in `DATA_DIR/compact4`.
- Outputs to `OUTPUT_DIR/<config name>/`,
  * where `<config name>.yml` determines the subdirectory name,
  * i.e., `unet1024/` and `vit4_1024/`.
  * model weights are `model<ifold>.pytorch`.

## Options

- The scripts accept `--settings=SETTINGS.json` for other settings files. 

- The output directory should not contain model weight files. Use `--overwrite` option to overwrite.
