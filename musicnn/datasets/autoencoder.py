import pkg_resources
import pickle as pkl

import numpy as np
import pandas as pd

from .audiodataset import AudioDataset
from .files import msd_lastfm50_splits


class MSDAudio(AudioDataset):
    """MSD Audio retrieval dataset

    Only retrieve audio (and potentially the last sample for WaveNet style learning)

    NOTE: it shares with same fold with MSD-LastFM50 dataset
    """
    def __init__(self, songs_root, split='train', fold=0,
                 crop_len=44100, transform=None):
        """"""
        split_info = msd_lastfm50_splits(fold, split)

        # call super class' constructor
        super().__init__(songs_root, split_info,
                         crop_len=crop_len, transform=transform)

    def _retrieve_target(self, fn):
        """"""
        return 0