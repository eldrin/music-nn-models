from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import VGGlike2DEncoder, Identity, STFT, StandardScaler
from ..datasets.files import stft_standard_scaler_stats


class VGGlike2DAutoTagger(nn.Module):
    """VGG-like 2D auto-tagger
    """
    def __init__(self, n_outputs, sig_len=44100, n_fft=1024, hop_sz=256,
                 n_hidden=256, layer1_channels=8, kernel_size=3,
                 pooling=nn.MaxPool2d, pool_size=2, non_linearity=nn.ReLU,
                 global_average_pooling=True, batch_norm=True, dropout=0.5):
        """"""
        super().__init__()

        if batch_norm:
            hid_bn = partial(nn.BatchNorm1d, num_features=n_hidden)
        else:
            hid_bn = Identity

        if dropout and isinstance(dropout, (int, float)) and dropout > 0:
            _dropout = partial(nn.Dropout, p=dropout)
        else:
            _dropout = Identity
        
        self.sclr = StandardScaler(*stft_standard_scaler_stats())
        self.stft = STFT(n_fft=n_fft, hop_sz=hop_sz,
                         magnitude=True, log=True)
        test_x = self.stft(torch.randn(sig_len))
        self.input_shape = test_x.numpy().shape  # shape of input STFT
        self.sig_len = sig_len  # length of input signal

        # initialize the encoder
        self.E = VGGlike2DEncoder(
            test_x.shape, n_hidden, layer1_channels,
            kernel_size, pooling, pool_size, non_linearity,
            global_average_pooling, batch_norm
        )

        # put on some decision (prediction) layers on top of it
        self.P = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            hid_bn(), non_linearity(), _dropout(),
            nn.Linear(n_hidden, n_outputs)
        )

    def preproc(self, x):
        return self.sclr(self.stft(x))[:, None]

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self.preproc(x), layer)

    def forward(self, x):
        return self.P(self.E(self.preproc(x)))