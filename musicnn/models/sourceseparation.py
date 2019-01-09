from functools import partial

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as f

from .layers import VGGlike2DEncoder, VGGlike2DDecoder, Identity, STFT
from .architecture import STFTInputNetwork
from ..config import Config as cfg


class VGGlike2DUNet(STFTInputNetwork):
    """VGG-like 2D segmentation (learning masking)
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
            global_average_pooling, batch_norm
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
        # self.iPa = nn.Sequential(
        #     nn.Linear(n_hidden, n_hidden),
        #     self.hid_bn(), non_linearity(), self._dropout()
        # )

        # put decoder for convolution encoders
        self.Dv = VGGlike2DDecoder(self.E)  # for vocals
        # self.Da = VGGlike2DDecoder(self.E)  # for accompaniments

    def get_hidden_state(self, x, layer=10):
        return self.E.get_hidden_state(self._preproc(x), layer)

    def get_bottleneck(self, x):
        return self.P(self.E(self._preproc(x)))

    def _preproc(self, x):
        X = torch.stft(x, self.stft.n_fft, self.stft.hop_sz,
                       window=self.stft.window)
        X = self.stft._magnitude(X)

        Z = self.sclr(self.stft._log(X))[:, None]
        return Z, X[:, None]

    def get_mask(self, X, z):
        Z = []  # for skip-connection

        for layer in self.E.encoder:
            z = layer(z)
            Z.append(z)

        # bottleneck
        z = self.P(z)
        zv = self.iPv(z)
        # zv, za = self.iPv(z), self.iPa(z)

        # Decoding
        # for iz, layer_v, layer_a in zip(Z[::-1], self.Dv.decoder, self.Da.decoder):
        #     zv = layer_v(zv + iz)
        #     za = layer_a(za + iz)

        # return zv, za  # input STFT, vocal STFT, accompaniment STFT

        for iz, layer_v in zip(Z[::-1], self.Dv.decoder):
            zv = layer_v(zv + iz)

        return torch.sigmoid(zv)

    def forward(self, x):
        # preprocessing
        z, X = self._preproc(x)  # scaled / not scaled

        # get mask
        M = self.get_mask(X, z)

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

        # # inverse scaling
        # Xm = self.sclr.inverse_transform(Xm[:, 0])
        Xm = Xm[:, 0]  # drop channel dim

        # to array
        if Xm.is_cuda:
            Xm = Xm.data.cpu().numpy()  # also remove channel dim
        else:
            Xm = Xm.data.numpy()

        # # to amplitude
        # Xm = librosa.db_to_amplitude(Xm)

        # to signal (n_batch, sig_len)
        x = np.array([librosa.istft(X) for X in Xm * np.exp(1j * Xp)])

        return x
