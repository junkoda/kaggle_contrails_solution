import numpy as np
import glob
import torch
import torch.nn as nn
import torchvision.transforms as T

di = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'


def load_dirs(data_type: str) -> np.ndarray[str]:
    """
    Returns list
      data directories
    """
    dirs = glob.glob('%s/%s/*' % (di, data_type))
    dirs.sort()
    dirs = np.array(dirs)

    return dirs


def load_data(path: str, t=4) -> dict:
    """
    Arg:
      path (str): data directory

    Returns: dict
      x (array): (B, 256, 256, T) for B=9 bands and T=8 timesteps
    """
    file_id = path.split('/')[-1]

    # Load input images
    bands = []
    for k in range(8, 17):
        a = np.load('%s/band_%02d.npy' % (path, k))
        bands.append(a)

    x = np.stack(bands, axis=0)  # (C, H, W, T)

    ret = {'file_id': file_id,
           'x': x[:, :, :, t]}
    return ret


# Ash-color false color
def rescale_range(x, f_min, f_max):
    # Rescale [f_min, f_max] to [0, 1]
    return (x - f_min) / (f_max - f_min)


def ash_color(x):
    """
    False color for contrail annotation
    x (array): (C, H, W) -> (3, H, W)
    """
    r = rescale_range(x[7] - x[6], -4, 2)
    g = rescale_range(x[6] - x[3], -4, 5)
    b = rescale_range(x[6], 243, 303)

    x = torch.stack([r, g, b], axis=0)
    x = 1 - x

    return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dirs, cfg):
        self.dirs = dirs
        nc = cfg['resize']
        self.resize = nn.Identity() if nc == 256 else T.Resize(nc, antialias=False)
        #self.normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        path = self.dirs[i]
        d = load_data(path)  # (C, H, W)

        # Ash-color image
        x = torch.from_numpy(d['x'])
        x = ash_color(x)  # (3, H, W)
        #x = self.normalize(x)

        # Resize image
        x = self.resize(x)

        ret = {'file_id': d['file_id'],
               'x': x}
        return ret


class Data:
    def __init__(self, data_type, *, debug=False):
        assert data_type in ['train', 'validation', 'test']

        dirs = load_dirs(data_type)
        if debug:
            dirs = dirs[:100]

        self.dirs = dirs

    def __len__(self):
        return len(self.dirs)

    def loader(self, cfg, m):
        batch_size = cfg['batch_size']

        return torch.utils.data.DataLoader(Dataset(self.dirs, m),
                                           batch_size=batch_size,
                                           num_workers=2)