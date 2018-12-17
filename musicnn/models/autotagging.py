from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import VGGlike2DEncoder, Identity


class VGGlike2DAutoTagger(nn.Module):
    """VGG-like 2D auto-tagger
    """
    def __init__(self, n_outputs, input_shape=(256, 512), n_hidden=256,
                 layer1_channels=8, kernel_size=3, pooling=nn.MaxPool2d,
                 pool_size=2, non_linearity=nn.ReLU,
                 global_average_pooling=True, batch_norm=True):
        """"""
        super().__init__()
        if batch_norm:
            hid_bn = partial(nn.BatchNorm1d, num_features=n_hidden)
        else:
            hid_bn = Identity

        # initialize the encoder
        self.E = VGGlike2DEncoder(
            input_shape, n_hidden, layer1_channels,
            kernel_size, pooling, pool_size, non_linearity,
            global_average_pooling, batch_norm
        )

        # put on some decision (prediction) layers on top of it
        self.P = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), hid_bn(), non_linearity(),
            nn.Linear(n_hidden, n_outputs)
        )
    
    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(x, layer)

    def forward(self, x):
        return self.P(self.E(x))