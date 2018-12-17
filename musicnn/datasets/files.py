import os
import glob

import pkg_resources


MSD_LASTFM50_METADATA = 'meta_data/msd_lastfm50_map.pkl'
MSD_LASTFM50_SPLIT = 'splits/msd_lastfm50_{}_{:d}.txt'


__all__ = ['msd_lastfm50_metadata', 'msd_lastfm50_splits']


def msd_lastfm50_metadata():
    """Get the path to the metadata for msd-lastfm-50

    Returns:
        str: filename of metadata
    """
    return pkg_resources.resource_filename(__name__, MSD_LASTFM50_METADATA)


def msd_lastfm50_splits(fold=0, split='train'):
    """Get the splits for msd-lastfm-50

    Returns:
        list: list conatins filenames of {train, valid, test} samples
    """
    assert split in {'train', 'valid', 'test'}
    return pkg_resources.resource_filename(
        __name__, MSD_LASTFM50_SPLIT.format(split, fold)
    )
