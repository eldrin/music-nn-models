from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import VGGlike2DEncoder, Identity
from .architecture import STFTInputNetwork


class VGGlike2DAutoTagger(STFTInputNetwork):
    """VGG-like 2D auto-tagger
    """
    def __init__(self, n_outputs, sig_len=44100, n_hidden=256, 
                 n_fft=1024, hop_sz=256, layer1_channels=8, kernel_size=3,
                 n_convs=1, pooling=nn.MaxPool2d, pool_size=2,
                 non_linearity=nn.ReLU, global_average_pooling=True,
                 batch_norm=True, dropout=0.5):
        """"""
        super().__init__(sig_len, n_hidden, batch_norm, dropout, n_fft, hop_sz,
                         magnitude=True, log=True, normalization='standard')

        # initialize the encoder
        self.E = VGGlike2DEncoder(
            self.input_shape, n_hidden, n_convs, layer1_channels,
            kernel_size, pooling, pool_size, non_linearity,
            global_average_pooling, batch_norm
        )

        # put on some decision (prediction) layers on top of it
        self.P = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            self.hid_bn(), non_linearity(), self._dropout(),
            nn.Linear(n_hidden, n_outputs)
        )

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        return self.P[:-1](self.E(self._preproc(x)))

    def forward(self, x):
        return self.P(self.E(self._preproc(x)))