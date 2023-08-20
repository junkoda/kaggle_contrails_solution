"""
Official code for encoding segmentation mask
https://www.kaggle.com/code/inversion/contrails-rle-submission
"""
import numpy as np
import pandas as pd


def to_str(x):
    """
    Converts list to a string representation
    Empty list returns '-'
    """
    if x:
        s = str(x).replace('[', '').replace(']', '').replace(',', '')
    else:
        s = '-'
    return s


def encode(x: np.ndarray, th: float, *, subsample=None) -> str:
    """
    Encode segmentation mask
    Arg:
      x (array[float]):  (H, W)
      subsample (Optional[float]): subsample positive pixels with probability in [0, 1]
    Return: run length encoding as list
    """
    dots = np.where(x.T.flatten() > th)[0]

    if subsample is not None:
        n = len(dots)
        idx = np.random.uniform(0, 1, n) < subsample
        dots = dots[idx]

    run_lengths = []  # list[(int, int) = [(ibegin, length), ...]
    prev = -2
    for b in dots:
        if b > prev + 1:
            # Add new segment
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return to_str(run_lengths)


def decode(mask_rle: str, shape=(256, 256)) -> np.ndarray:
    """
    Decode encoded str to array

    Args:
      mask_rle (str)
      shape: (H, W)

    Returns
       array[uint8]: (H, W)
    """
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if mask_rle != '-':
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape, order='F')


def write_submission(preds: list, th: float, ofilename: str, *, subsample=None):
    """
    preds (list):
      file_id (str)
      y_pred (array): (1, 256, 256)

    Returns df (pd.DataFrame)
      record_id, encoded_pixels
    """
    file_ids = []
    encoded_pixels = []
    for pred in preds:
        file_ids.append(pred['file_id'])
        encoded = encode(pred['y_pred'], th, subsample=subsample)
        encoded_pixels.append(encoded)

    df = pd.DataFrame({'record_id': file_ids,
                       'encoded_pixels': encoded_pixels})
    df.to_csv(ofilename, index=False)
    print(ofilename, 'written')
