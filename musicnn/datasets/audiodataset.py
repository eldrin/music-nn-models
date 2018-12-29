import os
from os.path import join, basename
import glob
from itertools import chain
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from ..utils.ops import mu_law_decode


class AudioDataset(Dataset):
    """"""
    def __init__(self, songs_root, subset_info, target=None,
                 crop_len=44100, transform=None, on_mem=False):
        """"""
        super().__init__()

        self.songs_root = songs_root
        with open(subset_info) as f:
            self.subset_fns = [l.replace('\n','') for l in f.readlines()]
        self.crop_len = crop_len
        self.transform = transform
        self.target = target
        self.on_mem = on_mem

        if self.on_mem:
            print('Loading audio to the memory!...')
            self.X = {}
            for fn in tqdm(self.subset_fns, ncols=80):
                self.X[fn] = np.load(join(self.songs_root, fn))

    def __len__(self):
        """"""
        return len(self.subset_fns)

    def __getitem__(self, idx):
        """"""
        # retrieve the track id from index
        fn = self.subset_fns[idx]

        # load the audio
        x = self._retrieve_audio(fn)
        x_ = np.array(self._crop_signal(x), dtype=x.dtype)

        # retrieve target
        y = self._retrieve_target(fn)

        # organize sample
        sample = {
            'signal': x_[..., :-1],
            'last': x_[..., -1].astype(int),
            'target': y
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _retrieve_audio(self, fn):
        """Retrieve audio signal from path

        Args:
            fn (str): path to the signal
        """
        if self.on_mem:
            return self.X[fn]
        else:
            return np.load(join(self.songs_root, fn), mmap_mode='r')

    def _retrieve_target(self, fn):
        """Retrieve target value from data
        """
        raise NotImplementedError()

    def _crop_signal(self, x):
        """"""
        if x.shape[-1] >= self.crop_len + 1:
            st = np.random.randint(x.shape[-1] - (self.crop_len + 1))
            return x[..., st: st + self.crop_len + 1]
        
        else:
            # zero_pad first
            x_ = np.zeros((self.crop_len + 1,), dtype=np.float32)
            st_ = np.random.randint((self.crop_len + 1) - x.shap[-1])
            x_[..., st_: st_ + x.shape[-1] + 1] = x
            return x_


class MuLawDecoding(object):
    """"""
    def __init__(self, quantization_channels=256, target_keys=['signal']):
        """"""
        self.target_keys = set(target_keys)
        self.quantization_channels = quantization_channels

    def __call__(self, sample):
        """"""
        transformed = {}

        # transform only target data
        for key in self.target_keys:
            transformed[key] = mu_law_decode(
                sample[key], self.quantization_channels
            ).numpy()

        # pass other data as they are
        for k, v in sample.items():
            if k not in target_keys:
                transformed[k] = v

        return transformed