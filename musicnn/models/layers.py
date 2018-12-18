import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """Base encoder class specifying the spec
    """
    def __init__(self):
        """"""
        super().__init__()

    def get_hidden_state(self, x, layer):
        """"""
        raise NotImplementedError()
    
    def forward(self, x):
        """"""
        raise NotImplementedError()


class VGGlike2DEncoder(BaseEncoder):
    """VGG-like model for encoding 2D music input to latent vector

    Args:
        n_hidden (int): number of hidden units of the output of the encoder
    """
    def __init__(self, input_shape=(256, 512), n_hidden=256, layer1_channels=8,
                 kernel_size=3, pooling=nn.MaxPool2d, pool_size=2,
                 non_linearity=nn.ReLU, global_average_pooling=True,
                 batch_norm=True):
        """"""
        super().__init__()

        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.pooling_layer = pooling
        self.non_linearity = non_linearity
        self.global_average_pooling = global_average_pooling
        self.conv_layers = []

        # build
        z = test_x = torch.randn(1, 1, input_shape[0], input_shape[1])
        in_chan = 1
        out_chan = layer1_channels
        while z.shape[-1] > 1 and z.shape[-2] > 1:
            # build
            self.conv_layers.extend([
                SameConv2D(in_chan, out_chan, kernel_size, batch_norm=batch_norm),
                self.pooling_layer(pool_size),
                self.non_linearity()
            ])
            encoder_ = nn.Sequential(*self.conv_layers)   
            z = encoder_(test_x)

            # update # of input channels
            if out_chan < n_hidden:
                in_chan = out_chan
                out_chan = 2 * in_chan
            else:
                in_chan = out_chan

        # global average pooling
        if self.global_average_pooling:
            self.conv_layers.append(GlobalAveragePooling())
        else:  # otherwise, flatten it
            self.conv_layers.append(Flattening())

        # initiate the encoder
        self.encoder = nn.ModuleList(self.conv_layers)

    def get_hidden_state(self, X, layer=10):
        z = X
        for layer in self.encoder[:layer]:
            z = layer(z)
        return z
    
    def forward(self, X):
        # check the size
        assert X.shape[1:] == (1, self.input_shape[0], self.input_shape[1])
        z = X
        for layer in self.encoder:
            z = layer(z)
        return z


class SameConv2D(nn.Module):
    """2-D convolution layer with same sized-output
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, batch_norm=True):

        super().__init__()

        # build padder
        pad = []
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        for size in kernel_size[::-1]:  # width first, height next
            pad.append(size // 2)
            if size % 2 == 0:  # even size
                pad.append(size // 2 - 1)
            else:  # odd size
                pad.append(size // 2)
        pad = tuple(pad)  # convert to tuple
        self.pad = nn.ReflectionPad2d(pad)

        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(self.pad(x))


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling
    """
    def __init__(self):
        super().__init__()
        self.flatten = Flattening()
    
    def forward(self, x):
        return self.flatten(x).mean(dim=-1)


class Flattening(nn.Module):
    """Flatten any spatial dimensions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)


class Identity(nn.Module):
    """Identity module
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    """STFT module
    """
    def __init__(self, n_fft, hop_sz, magnitude=True, log=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.magnitude = magnitude
        self.log = log

    def _magnitude(self, x):
        return (x[..., 0]**2 + x[..., 1]**2)**0.5
    
    def _log(self, x, eps=1e-8):
        eps = torch.tensor([eps]).float()
        if x.is_cuda: eps = eps.cuda()
        return torch.log10(torch.max(x, eps))
    
    def forward(self, x):
        X = torch.stft(x, self.n_fft, self.hop_sz)
        if self.magnitude:
            X = self._magnitude(X)
        if self.log:
            X = self._log(X)
        return X