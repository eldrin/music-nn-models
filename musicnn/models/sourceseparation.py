from functools import partial

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import (MFCCEncoder,
                     VGGlike2DEncoder,
                     VGGlike2DDecoder,
                     Identity,
                     STFT,
                     SameConv2D)
from .architecture import STFTInputNetwork
from ..config import Config as cfg


class VGGlike2DUNet(STFTInputNetwork):
    """VGG-like 2D segmentation (learning masking)
    """
    def __init__(self, sig_len=44100, n_hidden=256, n_fft=1024, hop_sz=256,
                 layer1_channels=8, kernel_size=3, n_convs=1, skip_conn=True,
                 pooling=nn.MaxPool2d, pool_size=2, non_linearity=nn.ReLU,
                 global_average_pooling=True, batch_norm=True, dropout=0.5,
                 normalization='standard'):
        """"""
        super().__init__(sig_len, n_hidden, batch_norm, dropout, n_fft, hop_sz,
                         magnitude=True, log=True, normalization=normalization)

        self.skip_conn = skip_conn

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
        self.iPv = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            self.hid_bn(), non_linearity(), self._dropout()
        )

        # put decoder for convolution encoders
        self.Dv = VGGlike2DDecoder(self.E)  # for vocals

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        # preprocessing
        _, z = self._preproc(x)  # scaled / not scaled

        for layer in self.E.encoder[:-1]:
            z, _ = layer(z)
        z = self.E.encoder[-1](z)

        # bottleneck
        z = self.P(z)
        return z

    def _preproc(self, x):
        X = torch.stft(x, self.stft.n_fft, self.stft.hop_sz,
                       window=self.stft.window)
        X = self.stft._magnitude(X)

        Z = self.sclr(self.stft._log(X))[:, None]
        return X[:, None], Z

    def get_mask(self, z):
        Z = []  # for skip-connection

        pool_idx = []
        for layer in self.E.encoder[:-1]:
            z, idx = layer(z)
            Z.append(z)
            pool_idx.append(idx)
        z = self.E.encoder[-1](z)

        # bottleneck
        z = self.P(z)
        zv = self.iPv(z)

        # Decoding
        zv = self.Dv.decoder[0](zv)
        for iz, idx, layer_v in zip(Z[::-1], pool_idx[::-1], self.Dv.decoder[1:]):
            if self.skip_conn:
                zv = layer_v(zv + iz, idx)
            else:
                zv = layer_v(zv, idx)

        return torch.sigmoid(zv)

    def forward(self, x):
        # preprocessing
        X, z = self._preproc(x)  # scaled / not scaled

        # get mask
        M = self.get_mask(z)

        return M * X

    def _post_process(self, Xm, Xp=None):
        """Inverse magnitude to signal

        NOTE: currently only support where original phase is provided

        Args:
            Xm (tensor): db-scale magnitude (output of forward of this model)
            Xp (numpy.ndarray): original phase

        Returns:
            x (numpy.ndarray): time-domain signal
        """
        assert Xp is not None

        # to array
        if Xm.is_cuda:
            Xm = Xm.data.cpu().numpy()  # also remove channel dim
        else:
            Xm = Xm.data.numpy()

        # to signal (n_batch, sig_len)
        x = np.array([librosa.istft(X) for X in Xm[0] * np.exp(1j * Xp)])

        return x


class MFCCAESourceSeparator(STFTInputNetwork):
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
        self.iPv = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            self.hid_bn(), non_linearity(), self._dropout()
        )

        # put decoder for convolution encoders
        self.Dv = VGGlike2DDecoder(_E)

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        # preprocessing
        X, _ = self._preproc(x)  # scaled / not scaled
        z = self.E(X)  # MFCC feature
        return self.P[0](z)

    def _preproc(self, x):
        X = torch.stft(x, self.stft.n_fft, self.stft.hop_sz,
                       window=self.stft.window)
        X = self.stft._magnitude(X)

        Z = self.sclr(self.stft._log(X))[:, None]
        return X[:, None], Z

    def get_mask(self, X):
        z = self.E(X)

        # bottleneck
        z = self.P(z)
        zv = self.iPv(z)

        # decoding
        zv = self.Dv(zv, [None] * len(self.Dv.decoder))
        return torch.sigmoid(zv)

    def forward(self, x):
        # preprocessing
        X, z = self._preproc(x)  # scaled / not scaled

        # get mask
        M = self.get_mask(X)

        return M * X

    def _post_process(self, Xm, Xp=None):
        """Inverse magnitude to signal

        NOTE: currently only support where original phase is provided

        Args:
            Xm (tensor): db-scale magnitude (output of forward of this model)
            Xp (numpy.ndarray): original phase

        Returns:
            x (numpy.ndarray): time-domain signal
        """
        assert Xp is not None

        # to array
        if Xm.is_cuda:
            Xm = Xm.data.cpu().numpy()  # also remove channel dim
        else:
            Xm = Xm.data.numpy()

        # to signal (n_batch, sig_len)
        x = np.array([librosa.istft(X) for X in Xm[0] * np.exp(1j * Xp)])

        return x
