
import numpy as np
import os
import time
import yaml
import argparse
import torch

from sklearn.model_selection import KFold
from contraillib import util
from data import Data
from model import Model


def compute(model, loader, device, *, th=0.1):
    """
    Compute score with threshold search
    """
    tb = time.time()

    was_training = model.training
    model.eval()

    y_trues = []
    y_preds = []

    for d in loader:
        x = d['x'].to(device)  # input image
        y = d['label']         # segmentation mask: (1, 256, 256)

        # Predict
        with torch.no_grad():
            y_sym_pred, y_pred = model(x)

        y_pred = y_pred.sigmoid()

        # Compute global dice
        assert y_pred.size(2) == 256
        y_pred = y_pred.cpu().numpy().flatten()
        y = y.numpy().flatten()

        idx = np.logical_or(y_pred > th, y == 1)

        y_trues.append(y[idx].copy())
        y_preds.append(y_pred[idx].copy())

    y = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)

    thresholds = np.arange(0.1, 0.6, 0.01)
    scores = []
    for th in thresholds:
        y_pred_th = y_pred > th
        dice = 2 * np.sum(y_pred_th * y) / (np.sum(y_pred_th) + np.sum(y))
        scores.append(dice)

    scores = np.array(scores)
    imax = np.argmax(scores)
    print('Score %.6f with th %.4f' % (scores[imax], thresholds[imax]))

    dt = time.time() - tb
    size = 4 * (len(y) + len(y_pred))
    print('%.1f min and %.1f MB' % (dt / 60, size / 1000**2))

    model.train(was_training)

    ret = {'score': scores[imax],
           'th': thresholds,
           'scores': scores,
           'dt': dt}
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    arg = parser.parse_args()

    device = torch.device('cuda')

    # Config yaml
    name = arg.name

    # Model directory
    exp_dir = '/kaggle/contrails/work/unet3/experiments/%s' % name
    if not os.path.exists(exp_dir):
        raise FileNotFoundError(exp_dir)

    # Config yaml
    filename = '%s/%s.yml' % (exp_dir, name)
    with open(filename, 'r') as f:
        cfg = yaml.safe_load(f)

    data = Data('train')
    print('Data', len(data))

    # Kfold
    nfolds = cfg['kfold']['k']
    folds = util.as_list(cfg['kfold']['folds'])  # list[int]
    kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)

    for ifold, (idx_train, idx_val) in enumerate(kfold.split(data.df)):
        if ifold not in folds:
            continue

        # Data
        loader_val = data.loader(idx_val, cfg, training=False, shuffle=True)

        # Model
        model = Model(cfg, pretrained=False)
        model_filename = '%s/model%d.pytorch' % (exp_dir, ifold)
        model.load_state_dict(torch.load(model_filename))
        model.eval()
        model.to(device)
        print('Load', model_filename)

        compute(model, loader_val, device)

        break
