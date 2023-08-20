Entry Points
============

This note explains how to train models. See also `winning_solution.md`.

# Assumptions

- `SETTINGS.json` exists in this directory.
  * different settings file can be specified with `--settings=filename` for python scripts

- Kaggle data are in `INPUT_DIR/google-research-identify-contrails-reduce-global-warming`
  * Put the data (or symbolic link to the data) in `./input` directory
  * or, set `INPUT_DIR` in `SETTINGS.json`
- `DATA_DIR` (default `./data`) exists and 70 GB of free disk space is available

# 1. Prepare

The training and validation data needs to be converted for efficient data loading.

```bash
$ python3 src/script/convert_data_compact4.py train
$ python3 src/script/convert_data_compact4.py validation
```

Output: <DATA_DIR>/compact4

# 2. Train

Test run

```
$ sh test_run.sh
```

Full training

```bash
$ sh train_all.sh
```

or,

```bash
$ python3 src/unet1024/evaluate.py src/unet1024/unet1024.yml 
$ python3 src/vit4/evaluate.py src/vit4/vit4_1024.yml
```

The script accepts `--settings=SETTINGS.json`.
