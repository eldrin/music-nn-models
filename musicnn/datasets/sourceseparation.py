import pkg_resources
import pickle as pkl

import numpy as np
import pandas as pd
import torch

from .audiodataset import AudioDataset
from ..datasets.files import musdb18_splits


class VocalSeparation(AudioDataset):
    """Vocal sepratation dataset from MUSDB18
    """
    def __init__(self, songs_root, split='train', fold=0,
                 crop_len=44100, transform=None, on_mem=False):
        """"""
        # load split info
        split_info = musdb18_splits(fold, split)

        # call super class' constructor
        super().__init__(songs_root, split_info,
                         target=None, crop_len=crop_len,
                         transform=transform, on_mem=on_mem)

    def __getitem__(self, idx):
        """"""
        # retrieve the track id from index
        fn = self.subset_fns[idx]

        # load the audio
        x = self._retrieve_audio(fn)

        # retrieve target
        y = self._retrieve_audio(fn.replace('mixture', 'vocals'))

        xy = torch.cat([x[None], y[None]], dim=0)

        xy_ = np.array(self._crop_signal(xy), dtype=xy.dtype)

        # organize sample
        sample = {
            'mixture': xy_[0, ..., :-1],
            'vocals': xy_[1, ..., :-1]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample