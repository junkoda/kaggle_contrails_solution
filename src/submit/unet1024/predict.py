import numpy as np
import os
import torch
from tqdm.auto import tqdm
from .data import Data
from .model import Model


def init(dirs):
    preds = []
    for d in dirs:
        file_id = os.path.basename(d)
        pred = {'file_id': file_id,
                'y_pred': np.zeros((1, 256, 256), dtype=np.float32)}
        preds.append(pred)

    return preds


def predict1(model, loader, w: float, device, preds: list):
    """
    Predict with one model
    """
    i = 0
    for d in tqdm(loader):
        x = d['x'].to(device)  # input image

        with torch.no_grad():
            _, y_pred = model(x)

        y_pred = y_pred.sigmoid().cpu().numpy()

        for y_pred1 in y_pred:
            preds[i]['y_pred'] += w * y_pred1
            i += 1


def run(data_type: str, cfg: dict, preds: list, *, debug=False):
    device = torch.device('cuda')

    # Data
    data = Data(data_type, debug=debug)

    if preds is None:
        preds = init(data.dirs)

    dataset_dir = cfg['input']['weight']

    for m in cfg['unet1024']['models']:
        """
        m (dict)
          name: resnest26
          encoder: timm-resnest26d
          folds: [0, 1, 2, 3, 4]
        """
        folds = m['folds']
        w = m['w'] / len(folds)

        loader = data.loader(cfg['unet1024'], m)

        for ifold in folds:
            # Load Model
            model_filename = '%s/unet1024/%s/model%d.pytorch' % (dataset_dir, m['name'], ifold)
            model = Model(m)
            model.load_state_dict(torch.load(model_filename))
            model.to(device)
            model.eval()
            print('Load %s %.4f' % (model_filename, w))

            # Predict
            predict1(model, loader, w, device, preds)

            del model

    return preds
