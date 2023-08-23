import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import albumentations as A
from grid import create_grid
from typing import Optional


# Augmentation
def augmentation(aug: str):
    if aug == 'd4':  # Dihedral group D4
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
        ])
    elif aug == 'rotation':
        return A.Compose([
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(rotate_limit=30, scale_limit=0.2, p=0.75)
        ])
    else:
        raise ValueError


# Ash-color false color
def rescale_range(x, f_min, f_max):
    # Rescale [f_min, f_max] to [0, 1]
    return (x - f_min) / (f_max - f_min)


def ash_color(x):
    """
    False color for contrail annotation
    x (array): (3, H, W) -> (3, H, W)
    """
    r = rescale_range(x[2] - x[1], -4, 2)  # ch 7 - 6
    g = rescale_range(x[1] - x[0], -4, 5)  # ch 6 - 3
    b = rescale_range(x[1], 243, 303)      # ch 6

    x = torch.stack([r, g, b], axis=0)
    x = 1 - x  # does not matter. I drop clip [0, 1] but do not see difference.

    return x


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for DataLoader
    d: dict = dataset[i]
    """
    def __init__(self, df: pd.DataFrame, cfg: dict, *, augment=False):
        self.df = df
        self.augment = None
        if augment and cfg['data']['augment']:
            self.augment = augmentation(cfg['data']['augment'])

        self.annotation_mean = cfg['data']['annotation_mean']
        assert self.annotation_mean in ['mix', True, False]

        # Number of pixels per dimension for input x and symmetric target y_sym (256 or 512)
        nc = cfg['data']['resize']

        # Resize input
        self.resize = nn.Identity() if nc == 256 else T.Resize(nc, antialias=False)

        # Sample y_sym from y with this grid
        self.grid = create_grid(nc, offset=0.5)

        self.y_sym_mode = cfg['data']['y_sym_mode']
        assert self.y_sym_mode in ['bilinear', 'nearest']

        # Probability of augmentation (used 0.95)
        # Since asym_conv is trained with 1 - augment_prob, do not set this to 1.
        self.augment_prob = cfg['data']['augment_prob']

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        filename = r['filename']

        # Load
        with h5py.File(filename, 'r') as f:
            # h5 file contains t=1,2,3,4, and index 3 is t=4
            x = f['x'][3, :]  # (3, H, W)

            # Target
            if 'annotation_mean' in f:
                if self.annotation_mean == 'mix':
                    y = 0.5 * (f['annotation_mean'][:].astype(np.float32) + f['y'][:].astype(np.float32))
                elif self.annotation_mean is True:
                    y = f['annotation_mean'][:].astype(np.float32)
                else:
                    y = f['y'][:].astype(np.float32)  # (1, H, W)
            else:
                y = None

            # Ground truth for validation
            label = torch.from_numpy(f['y'][:].astype(np.float32))

        x = torch.from_numpy(x)
        if y is not None:
            y = torch.from_numpy(y)

        # Create color image
        x = ash_color(x)
        x = self.resize(x)

        # Sample shifted y_sym from y
        if y is not None:
            y_sym = F.grid_sample(y.unsqueeze(0), self.grid,
                                  mode=self.y_sym_mode, padding_mode='border',
                                  align_corners=False).squeeze(0)

        # Augment
        w_original = 1.0
        if self.augment is not None and np.random.random() < self.augment_prob:
            w_original = 0.0

            x = x.permute(1, 2, 0).numpy()  # => (H, W, C)
            y_sym = y_sym.permute(1, 2, 0).numpy()

            aug = self.augment(image=x, mask=y_sym)

            x = torch.from_numpy(aug['image'].transpose(2, 0, 1))     # => (C, H, W)
            y_sym = torch.from_numpy(aug['mask'].transpose(2, 0, 1))  # array (1, 256, 256)

        # Return value
        d = {'x': x,
             'w': np.float32(w_original)}  # 1 if y is original not augmented

        # y is soft label for loss
        if y is not None:
            d['y_sym'] = y_sym
            d['y'] = y

        # Label is ground truth for score
        if label is not None:
            d['label'] = label
        return d


class Data:
    """
    Wrapper class for Dataset and DataLoader
    """
    def __init__(self, data_type: str, data_dir, *, debug=False):
        """
        Args:
          data_type (str): train or validation
          data_dir (str):  DATA_DIR in SETTINGS.json
          debug (bool): Use small subset of data if True

        Input:
          data_dir/compact4/<data_type>/*.h5, using filenames in,
          data_dir/<data_type>.csv
        """
        # Load filename list
        df = pd.read_csv('%s/%s.csv' % (data_dir, data_type))
        if debug:
            df = df.iloc[:100]

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def dataset(self, idx: Optional[np.ndarray], cfg: dict, augment: bool, *, positive_only=False):
        """
        Return Dataset for DataLoader
        """
        df = self.df.iloc[idx] if idx is not None else self.df

        if positive_only:
            df = df[df.label_sum > 0]
            print('Data positive only: %d' % len(df))

        return Dataset(df, cfg, augment=augment)

    def loader(self, idx: Optional[np.ndarray], cfg: dict, *, augment=False, shuffle=False, drop_last=False):
        """
        Return DataLoader for minibatch loop

        Args:
          idx (Optional[array]): indices for subset, e.g., idx_train, idx_train from kfold.split(),
                                 None for all data.
          cfg (dict): config for loader
          augment (bool): apply augmentation (True for training, False for validataion)
          suffle, drop_last (bool): options for DataLoader
        """
        batch_size = cfg['train']['batch_size']
        num_workers = cfg['train']['num_workers']
        positive_only = cfg['data']['positive_only']  # Train with positive data only (at least one positive pixel)

        ds = self.dataset(idx, cfg, augment, positive_only=positive_only)
        return torch.utils.data.DataLoader(ds,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=shuffle,
                                           drop_last=drop_last)
