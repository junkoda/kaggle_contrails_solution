"""
Compute scores for a range of thresholds and print best score.
"""
import numpy as np
import time
import torch


def compute(model, loader, device, *, th=0.1):
    """
    Compute score with threshold search

    Args:
      loader: PyTorch DataLoader
      device: PyTorch device
      th (float): small enough threshold neglecting below
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

        # Collect positive candidates for efficient threshold search
        idx = np.logical_or(y_pred > th, y == 1)

        y_trues.append(y[idx].copy())
        y_preds.append(y_pred[idx].copy())

    # 1-dimensional array
    y = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)

    # Compute scores for a range of thresholds
    thresholds = np.arange(0.1, 0.6, 0.01)
    scores = []
    for th in thresholds:
        y_pred_th = y_pred > th
        dice = 2 * np.sum(y_pred_th * y) / (np.sum(y_pred_th) + np.sum(y))
        scores.append(dice)

    # Best threshold
    scores = np.array(scores)
    imax = np.argmax(scores)
    print('Score %.6f with th %.4f' % (scores[imax], thresholds[imax]))

    dt = time.time() - tb
    size = 4 * (len(y) + len(y_pred))
    print('%.1f min and %.1f MB' % (dt / 60, size / 1000**2))

    model.train(was_training)

    # Return values
    ret = {'score': scores[imax],
           'th': thresholds,
           'scores': scores,
           'dt': dt}
    return ret
