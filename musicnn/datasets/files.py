import os
import glob
import pkg_resources

import numpy as np


MSD_LASTFM50_LABEL = 'meta_data/msd_lastfm50_map.pkl'
MSD_LASTFM50_SPLIT = 'splits/msd_lastfm50_{}_{:d}.txt'
IRMAS_TRAIN_METADATA = 'meta_data/irmas_train_fn_songid_map.pkl'
IRMAS_TRAIN_SPLIT = 'splits/irmas_trn_{}_{:d}.txt'
STANDARD_SCALER_STATS = 'data/dBstft_standardscaler_stats.npy'


__all__ = [
    'msd_lastfm50_label',
    'msd_lastfm50_splits',
    'irmas_train_metadata',
    'irmas_train_splits',
    'stft_standard_scaler_stats'
]


def msd_lastfm50_label():
    """Get the path to the label data for msd-lastfm-50

    Returns:
        str: filename of metadata
    """
    return pkg_resources.resource_filename(__name__, MSD_LASTFM50_LABEL)


def msd_lastfm50_splits(fold=0, split='train'):
    """Get the splits for msd-lastfm-50

    Returns:
        list: list conatins filenames of {train, valid, test} samples
    """
    assert split in {'train', 'valid', 'test'}
    return pkg_resources.resource_filename(
        __name__, MSD_LASTFM50_SPLIT.format(split, fold)
    )


def stft_standard_scaler_stats():
    """Get the path to the stat (mean, std) data for the dB-Spectrum

    Returns:
        numpy.ndarray: mean of dB spectrum 
        numpy.ndarray: std of dB spectrum
    """
    fn = pkg_resources.resource_filename(__name__, STANDARD_SCALER_STATS)
    stats = np.load(fn)
    return stats[:, 0], stats[:, 1]


def irmas_train_metadata():
    """Get the path to the metadata (fn-songid map) for IRMAS training set

    Returns:
        str: filename of metadata
    """
    return pkg_resources.resource_filename(__name__, IRMAS_TRAIN_METADATA)


def irmas_train_splits(fold=0, split='train'):
    """Get the splits for msd-lastfm-50

    Returns:
        list: list conatins filenames of {train, valid, test} samples
    """
    assert split in {'train', 'valid', 'test'}
    return pkg_resources.resource_filename(
        __name__, IRMAS_TRAIN_SPLIT.format(split, fold)
    )