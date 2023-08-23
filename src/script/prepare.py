"""
Convert data to HDF5 format for efficient data loading

- x
  * bands 11, 14, 15 for ash color
  * t=1,2,3,4
- y
  * annotator mean (soft label)
  * label (ground truth for score)

Prerequisites
- INPUT_DIR and DATA_DIR in SETTINGS.json
- Kaggle competition data in <INPUT_DIR>/google-research-identify-contrails-reduce-global-warming
- <DATA_DIR> directory for output

Command
$ python3 src/script/convert_data_compact4.py train
$ python3 src/script/convert_data_compact4.py validation

Output
- <DATA_DIR>/compact4/train/<file_ids>.h5, validation/<file_ids>.h5

  x (array[float32]):      image (T, C, H, W) for T=4 timesteps, C=3 bands, and H=W=256
  y (array[float16]):      label (1, H, W)
  annotation_mean (array[float16]): mean of individual annotations (1, H, W)
  label_sum (int):         sum of y

To use HDF5 file:
$ pip install h5py
"""
import numpy as np
import pandas as pd
import os
import glob
import h5py
import json
import argparse
from tqdm.auto import tqdm

competition = 'google-research-identify-contrails-reduce-global-warming'


def convert(data_type: str, input_dir: str, data_dir: str) -> None:
    """
    Convert original npy data to h5 files

    Args:
      data_type (str): train or validation
      input_dir (str): INPUT_DIR in SETTINGS.json
      data_dir (str):  output directory DATA_DIR in SETTINGS.json

    Input:
      <INPUT_DIR>/google-research-identify-contrails-reduce-global-warming/<data_type>
    Output:
      <DATA_DIR>/compact4/<data_type>/<file_id>.h5
    """
    # Output directory
    odir = '%s/compact4/%s' % (data_dir, data_type)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # Original Kaggle data files
    dirs = glob.glob('%s/%s/%s/*' % (input_dir, competition, data_type))
    dirs.sort()
    print('Data', len(dirs))
    if not dirs:
        print('No files found in %s/%s' % (input_dir, competition))
        raise FileNotFoundError

    # Convert to HDF files
    for path in tqdm(dirs, desc=data_type, ncols=78):
        file_id = path.split('/')[-1]

        # Load input images
        bands = []
        for k in [11, 14, 15]:
            a = np.load('%s/band_%02d.npy' % (path, k))[:, :, 1:5]  # (256, 256, T=4)
            bands.append(a)

        x = np.stack(bands, axis=0)  # (3, H, W, T)
        x = x.transpose(3, 0, 1, 2)  # (T, C, H, W)

        # Ground truth label
        y = np.load('%s/human_pixel_masks.npy' % path)
        y_sum = np.sum(y)
        y = y.reshape(1, 256, 256)

        # Load inidividual masks
        if data_type == 'train':
            # Mean of individual annotations
            annot = np.load('%s/human_individual_masks.npy' % path)
            annot = annot.astype(np.float32)  # (256, 256, 1, A)
            annot = np.mean(annot, axis=3)
            annot = annot.reshape(1, 256, 256)

        ofilename = '%s/%s.h5' % (odir, file_id)
        with h5py.File(ofilename, 'w') as f:
            f['x'] = x  # => (T, C, H, W) for H=W=256
            f['y'] = y.astype(np.float16)
            f['label_sum'] = y_sum
            if data_type == 'train':
                f['annotation_mean'] = annot.astype(np.float16)

    print(odir, 'written', len(dirs))


def create_df(data_type: str, data_dir: str) -> None:
    """
    Create and save DataFrame of data file names

    Args:
      data_type (str): train or validation
      data_dir (str):  output directory; DATA_DIR in SETTINGS.json
    
    Input:
      <DATA_DIR>/compact4/<data_type>/*.h5
    Output:
      <DATA_DIR>/<data_type>.csv
    """
    assert data_type in ['train', 'validation']
    filenames = glob.glob('%s/compact4/%s/*.h5' % (data_dir, data_type))
    filenames.sort()
    assert filenames

    file_ids = []
    label_sums = []
    for filename in filenames:
        file_id = os.path.basename(filename).replace('.h5', '')
        file_ids.append(file_id)
        with h5py.File(filename, 'r') as f:
            label_sum = f['label_sum'][()]
            label_sums.append(label_sum)

    df = pd.DataFrame({'file_id': file_ids, 'filename': filenames, 'label_sum': label_sums})

    ofilename = '%s/%s.csv' % (data_dir, data_type)
    df.to_csv(ofilename, index=False)
    print(ofilename, 'written', len(df))


def main() -> None:
    # Command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument('data_type', help='train or validation')
    parser.add_argument('--settings', default='SETTINGS.json', help='path config file')
    arg = parser.parse_args()
    data_type = arg.data_type

    assert data_type in ['train', 'validation']

    # SETTINGS.json
    with open(arg.settings) as f:
        settings = json.load(f)

    # Output directory
    data_dir = settings['DATA_DIR']
    if not os.path.exists(data_dir):
        print('Data directory %s does not exist' % data_dir)
        raise FileNotFoundError

    # Input dir
    input_dir = settings['INPUT_DIR']

    # Convert npy -> h5
    convert(data_type, input_dir, data_dir)

    # Write file list
    create_df(data_type, data_dir)


if __name__ == '__main__':
    main()
