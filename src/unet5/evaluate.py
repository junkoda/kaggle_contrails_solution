"""
U-Net with single timestep 3-channel image
"""
import numpy as np
import os
import glob
import json
import time
import yaml
import pickle
import signal
import argparse

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torchvision.transforms
import segmentation_models_pytorch as smp

from data import Data
from model import Model
from lr_scheduler import Scheduler
import util
import score
import losses


signal.signal(signal.SIGINT, signal.SIG_DFL)
torch.set_num_threads(1)
device = torch.device('cuda')
resize = torchvision.transforms.Resize(256, antialias=False)


def evaluate(model, loader_val, *, th=0.45):
    """
    Compute validation loss and score
    """
    tb = time.time()

    was_training = model.training
    model.eval()

    n_sum = 0
    loss_sum = 0.0
    dice_sum = 0.0
    tp = 0
    positives_pred = 0
    positives_true = 0

    for d in loader_val:
        x = d['x'].to(device)        # input image
        if 'y' in d:
            y = d['y'].to(device)    # segmentation mask: 1 x H x W; upscaled soft label
            y_sym = d['y_sym'].to(device)
        else:
            y = None

        # label is ground-truth segmentation mask for score (always 256 x 256 binary)
        label = d['label'].to(device)
        batch_size = len(x)

        # Predict
        with torch.no_grad():
            y_sym_pred, y_pred = model(x)  # (batch_size, 1, H, W)

        # Compute loss
        if y is not None:
            w = 1 - augment_fraction  # non-augmented fraction in training
            loss = criterion(y_sym_pred, y_sym, y_pred, y, w)
            loss_dice = dice(y_pred, y)

            n_sum += batch_size
            loss_sum += loss.item() * batch_size
            dice_sum += loss_dice.item() * batch_size

        # Compute score
        y_pred = y_pred.sigmoid() > th
        tp += (y_pred * label).sum().item()
        positives_pred += y_pred.sum().item()
        positives_true += label.sum().item()

    global_dice = 2 * tp / (positives_pred + positives_true)

    model.train(was_training)

    dt = time.time() - tb
    ret = {'score': global_dice,
           'dt': dt}

    if y is not None:
        ret['loss'] = loss_sum / n_sum
        ret['dice'] = 1 - dice_sum / n_sum
    return ret


#
# Main
#
# Command-line options
parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--settings', default='SETTINGS.json', help='path config file')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--debug', action='store_true')
arg = parser.parse_args()

debug = arg.debug

# SETTINGS.json
with open(arg.settings) as f:
    settings = json.load(f)

# Config yaml
name = 'debug' if arg.debug else \
        os.path.basename(arg.filename).replace('.yml', '')
with open(arg.filename, 'r') as f:
    cfg = yaml.safe_load(f)

# Output directory
odir = settings['OUTPUT_DIR']
if not os.path.exists(odir):
    raise FileNotFoundError(odir)

odir = '%s/%s' % (odir, name)
if os.path.exists(odir):
    if (not arg.overwrite) and (not debug):
        if glob.glob('%s/model*.pytorch' % odir):
            raise FileExistsError(odir)
else:
    os.mkdir(odir)
    print(odir, 'created')

# Data
data_dir = settings['DATA_DIR']
data = Data('train', data_dir, debug=debug)
data_test = Data('validation', data_dir, debug=debug)
print('Data', len(data), len(data_test))

# Kfold
nfolds = cfg['kfold']['k']
folds = util.as_list(cfg['kfold']['folds'])  # list[int]
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=42)
print('folds', folds, '/', nfolds)

# Loss
loss_type = cfg['train']['loss']
criterion = losses.BCELoss()

# Dice score for evaluation metric
dice = smp.losses.DiceLoss('binary', from_logits=True)

weight_decay = float(cfg['train']['weight_decay'])
print('weight_decay', weight_decay)

augment_fraction = cfg['data']['augment_prob']
steps_per_epoch = cfg['val']['per_epoch']  # number of validation per epoch (int >= 1)
th_val = cfg['val']['th']     # threshold for oof val score
th_test = cfg['test']['th']   # for validation score
print('val threshold %.2f %.2f' % (th_val, th_test))

#
# Train loop
#
log = {}
epochs_log = []
losses_train = []
losses_val = []
final_scores = []

n_sum = 0
loss_sum = 0.0
lrs = []

tb_global = time.time()
for ifold, (idx_train, idx_val) in enumerate(kfold.split(data.df)):
    if ifold not in folds:
        continue

    # Data
    loader_train = data.loader(idx_train, cfg, augment=True, drop_last=True, shuffle=True)
    loader_val = data.loader(idx_val, cfg)
    loader_test = data_test.loader(None, cfg)

    nbatch = len(loader_train)

    # Model
    model = Model(cfg, pretrained=True)
    if cfg['model']['load'] is not None:
        model_filename = cfg['model']['load']
        model.load_state_dict(torch.load(model_filename))
        print('Load', model_filename)

    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=weight_decay)

    scheduler = Scheduler(optimizer, cfg['scheduler'])
    epochs = 4 if debug else len(scheduler)
    print('%d epochs' % epochs)

    tb = time.time()
    dt_val = 0.0
    print('KFold %d/%d' % (ifold, nfolds))
    print('Epoch        loss          dice  score         lr       time')
    for iepoch in range(epochs):
        icheck = [nbatch * (i + 1) // steps_per_epoch - 1 for i in range(steps_per_epoch)]

        for ibatch, d in enumerate(loader_train):
            x = d['x'].to(device)          # input image
            y = d['y'].to(device)          # segmentation label
            y_sym = d['y_sym'].to(device)  # symmetric label
            w = d['w'].to(device)          # w=1 if x is not augmented
            batch_size = len(x)

            optimizer.zero_grad()

            # Predict
            y_sym_pred, y_pred = model(x)  # (batch_size, 1, 256, 256)
            loss = criterion(y_sym_pred, y_sym, y_pred, y, w)

            # Backpropagate
            loss.backward()

            n_sum += batch_size
            loss_sum += batch_size * loss.item()

            nn.utils.clip_grad_value_(model.parameters(), 1000.0)
            optimizer.step()

            ep = iepoch + (ibatch + 1) / nbatch
            lr = optimizer.param_groups[0]['lr']
            lrs.append((ep, lr))

            # Validation
            if ibatch == icheck[0]:
                icheck.pop(0)

                epochs_log.append(iepoch + (ibatch + 1) / nbatch)
                loss_train = loss_sum / n_sum
                losses_train.append(loss_train)

                val = evaluate(model, loader_val, th=th_val)
                test = evaluate(model, loader_test, th=th_test)

                losses_val.append(val['loss'])
                dt = time.time() - tb
                dt_val += val['dt'] + test['dt']

                print('Epoch %5.2f %6.3f %6.3f  %.3f %.3f %.3f  %5.1e %5.1f %5.1f min' % (ep,
                      10 * loss_train, 10 * val['loss'],
                      val['dice'], val['score'], test['score'],
                      lr, dt_val / 60, dt / 60))

                # Reset train loss
                n_sum = 0
                loss_sum = 0.0

            scheduler.step(ep)

        # Epoch done
    # Training done

    # Final score
    sc = score.compute(model, loader_test, device)
    final_scores.append(sc['score'])

    # Save model
    model.eval()
    model.to('cpu')
    ofilename = '%s/model%d.pytorch' % (odir, ifold)
    torch.save(model.state_dict(), ofilename)
    print(ofilename, 'written')

    # Save log
    dt = time.time() - tb
    log['fold%d' % ifold] = {
        'epoch': np.array(epochs_log),
        'lr': np.array(lrs),
        'loss_train': np.array(losses_train),
        'loss_val': np.array(losses_val),
        'time': {'total': dt, 'val': dt_val},
        'score': sc
    }

    del model

# Kfolds done
dt = time.time() - tb_global
print('Total time: %.2f min' % (dt / 60))

# Score
if len(final_scores) >= 4:
    print('Final score: %.6f ± %.6f' % (np.mean(final_scores), np.std(final_scores)))

# Write log
ofilename = '%s/log.pkl' % odir
with open(ofilename, 'wb') as f:
    pickle.dump(log, f)
