from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import MFCCEncoder, VGGlike2DEncoder, VGGlike2DDecoder, Identity
from .architecture import STFTInputNetwork


class VGGlike2DAutoEncoder(STFTInputNetwork):
    """VGG-like 2D auto-encoder
    """
    def __init__(self, sig_len=44100, n_hidden=256, n_fft=1024, hop_sz=256,
                 layer1_channels=8, kernel_size=3, n_convs=1,
                 pooling=nn.MaxPool2d, pool_size=2, non_linearity=nn.ReLU,
                 global_average_pooling=True, batch_norm=True, dropout=0.5,
                 normalization='standard'):
        """"""
        super().__init__(sig_len, n_hidden, batch_norm, dropout, n_fft, hop_sz,
                         magnitude=True, log=True, normalization=normalization)

        # initialize the encoder
        self.E = VGGlike2DEncoder(
            self.input_shape, n_hidden, n_convs, layer1_channels,
            kernel_size, pooling, pool_size, non_linearity,
            global_average_pooling, batch_norm, rtn_pool_idx=True
        )

        # put on some decision (prediction) layers on top of it
        self.P = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            self.hid_bn(), non_linearity(), self._dropout(),
        )

        # put decoder on top of it
        self.iP = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            self.hid_bn(), non_linearity(), self._dropout()
        )

        # put decoder for convolution encoders
        self.D = VGGlike2DDecoder(self.E)

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        X = self._preproc(x)
        z, _ = self.E(X)
        return self.P(z)

    def forward(self, x):
        X = self._preproc(x)
        z, pool_idx = self.E(X)
        Xhat = self.D(self.iP(self.P(z)), pool_idx)
        return X, Xhat


class MFCCAutoEncoder(STFTInputNetwork):
    """2D auto-tagger with fully on-line MFCC encoder
    """
    def __init__(self, sig_len=44100, n_mfccs=25, sr=22050,
                 n_fft=1024, hop_sz=256, non_linearity=nn.ReLU,
                 batch_norm=True, layer1_channels=8, dropout=0.5):
        """"""
        self.n_hidden = (n_mfccs - 1) * 6  # with stats and deltas

        super().__init__(sig_len, self.n_hidden, batch_norm, dropout,
                         n_fft, hop_sz, magnitude=True, log=False,
                         normalization=False)

        # initialize the encoder
        self.E = MFCCEncoder(n_mfccs, n_fft, sr, include_coeff0=False)


        # initialize the encoder
        _E = VGGlike2DEncoder(
            self.input_shape, self.n_hidden,
            n_convs=1, layer1_channels=layer1_channels,
            kernel_size=3, pooling=nn.AvgPool2d,
            pool_size=2, non_linearity=non_linearity,
            global_average_pooling=True,
            batch_norm=batch_norm, rtn_pool_idx=False
        )

        # put on some decision (prediction) layers on top of it
        self.P = nn.Sequential(
            self.hid_bn(), non_linearity(), self._dropout(),
        )

        # put decoder on top of it
        self.iP = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            self.hid_bn(), non_linearity(), self._dropout()
        )

        # put decoder for convolution encoders
        self.D = VGGlike2DDecoder(_E)

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        X = self._preproc(x)
        z = self.E(X)
        return z

    def forward(self, x):
        X = self._preproc(x)
        z = self.E(X)
        z = self.iP(self.P(z))
        Xhat = self.D(z, [None] * len(self.D.decoder))
        return X, Xhat

