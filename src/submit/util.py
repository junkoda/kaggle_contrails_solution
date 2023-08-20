
import numpy as np
import os
import h5py
import glob

di = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'


def write_preds(preds: list, odir: str):
    if not os.path.exists('predict'):
        raise FileNotFoundError

    odir = 'predict/%s' % odir
    if not os.path.exists(odir):
        os.mkdir(odir)

    for pred in preds:
        file_id = pred['file_id']
        ofilename = '%s/%s.h5' % (odir, file_id)

        with h5py.File(ofilename, 'w') as f:
            f['y_pred'] = pred['y_pred'].astype(np.float16)

    print(odir, 'written')


def load_preds(odir):
    filenames = glob.glob('%s/*.h5' % odir)
    filenames.sort()
    if not filenames:
        raise FileNotFoundError

    preds = []
    for filename in filenames:
        file_id = os.path.basename(filename).replace('.h5', '')
        with h5py.File(filename, 'r') as f:
            y_pred = f['y_pred'][:]
        
        pred = {'file_id': file_id,
                'y_pred': y_pred}
        preds.append(pred)

    return preds

def load_label(preds):
    for pred in preds:
        file_id = pred['file_id']


def check_preds_finite(preds):
    """
    Assert y_pred values are all finite
    """
    for pred in preds:
        y_pred = pred['y_pred']
        assert np.isfinite(y_pred).all()
        
