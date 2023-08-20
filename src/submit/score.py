import numpy as np
import pandas as pd
import argparse
from submit import decode

di = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'


def load_label(data_type: str, file_id: str):
    """
    Returns ground-truth mask
      array[int]: (256, 256)
    """
    filename = '%s/%s/%s/human_pixel_masks.npy' % (di, data_type, file_id)
    y = np.load(filename)  # array[int32] (256, 256, 1)
    return y.reshape(256, 256)


def compute_score(data_type, submit):
    """
    Returns:
      tp (int): number of true positives
      fp (int): false positives
      fn (int): false negatives
    """
    assert data_type in ['train', 'validation']

    tp = 0
    pos_pred = 0
    pos_true = 0
    for i, r in submit.iterrows():
        file_id = r['record_id']
        y_pred_str = r['encoded_pixels']
        y_pred = decode(y_pred_str)
        y = load_label(data_type, file_id)

        tp += np.sum(y * y_pred)
        pos_pred += np.sum(y_pred)
        pos_true += np.sum(y)

    score = 2 * tp / (pos_pred + pos_true)
    print('score %.6f / %d' % (score, len(submit)))
    print('tp %d, fp %d, fn %d' % (tp, pos_pred - tp, pos_true - tp))

    ret = {'score': score,
           'tp': tp,
           'fp': pos_pred - tp,
           'fn': pos_true - tp}
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--data-type', default='validation')
    arg = parser.parse_args()

    submit = pd.read_csv(arg.filename)
    compute_score(arg.data_type, submit)


if __name__ == '__main__':
    main()
