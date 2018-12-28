import os
from os.path import join, dirname, basename
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
import pickle as pkl
import random

from musicnn.datasets.files import irmas_train_metadata


RATIO = (0.8, 0.1, 0.1)


def load_irmas_metadata():
    """Load IRMAS metadata from repo

    Returns:
        list of tuple: contains (fn ('.wav'), songid) pair for all files
    """
    return pkl.load(open(irmas_train_metadata(), 'rb'))


def get_split_indice(n, ratio=RATIO):
    """Get random split with given train/valid/test ratio

    Args:
        n (int): number of samples of given data
        ratio (tuple of float): contains ratio between train/valid/test

    Returns:
        tuple of numpy.ndarray: contains the random index of items
                                based on the ratio
    """
    rnd_idx = list(range(n))
    random.shuffle(rnd_idx)
    train_bound = int(n * ratio[0])
    valid_bound = int(n * (ratio[0] + ratio[1]))
    return (
        rnd_idx[:train_bound],
        rnd_idx[train_bound:valid_bound],
        rnd_idx[valid_bound:]
    )


def get_split_filenames(ratio=RATIO):
    """Get split filenames

    Args:
        ratio (tuple of float): contains ratio between train/valid/test

    Returns:
        tuple of filenames for train / valid / test set
    """
    fn_songids = load_irmas_metadata()
    uniq_songs = list(set([x[1] for x in fn_songids.items()]))
    [train, valid, test] = [
        [fn[0] for fn in fn_songids.items() if fn[1] in split_]
        for split_
        in [
            [uniq_songs[j] for j in split_index]
            for split_index in get_split_indice(len(uniq_songs), ratio)
        ]
    ]
    return train, valid, test

    
def write_result(path, split=0):
    """Write split result to the file

    Args:
        path (str): path to save output split doc (.txt)
        split (int): indication of which split is
    """
    splits = get_split_filenames()
    for split_name, split_fns in zip(['train', 'valid', 'test'], splits):
        out_fn = join(path, 'irmas_trn_{}_{:d}.txt'.format(split_name, split))
        with open(out_fn, 'w') as f:
            [
                f.write('{}.npy\n'.format(fn.split('.')[0]))
                for fn in split_fns
            ]


if __name__ == "__main__":
    write_result(
        path=join(dirname(__file__), '../../musicnn/datasets/splits/'),
        split=0
    )