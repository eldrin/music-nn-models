import warnings 
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import STFT, StandardScaler, Identity, SumToOneNormalization
from ..datasets.files import stft_standard_scaler_stats


class BaseArchitecture(nn.Module):
    """Base Architecture for extension
    """
    def __init__(self, n_hidden, batch_norm, dropout):
        super().__init__()

        self.n_hidden = n_hidden

        if batch_norm:
            self.hid_bn = partial(nn.BatchNorm1d, num_features=n_hidden)
        else:
            self.hid_bn = Identity

        if dropout and isinstance(dropout, (int, float)) and dropout > 0:
            self._dropout = partial(nn.Dropout, p=dropout)
        else:
            self._dropout = Identity


class STFTInputNetwork(BaseArchitecture):
    """An architecture takes STFT as input representation

    Args:
        sig_len (int): input signal (1d) length
        n_fft (int): number of FFT points
        hop_sz (int): hop size for sliding windowing
        magnitude (bool): take the magnitude of the spectra
        log (bool): take the db-scale spectra
        normalization (str, None): standardizing method {'standard', 'sum2one', None}
    """
    def __init__(self, sig_len, n_hidden, batch_norm, dropout, n_fft, hop_sz,
                 magnitude=True, log=True, normalization='standard'):
        super().__init__(n_hidden, batch_norm, dropout)
        self.sig_len = sig_len
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.magnitude = magnitude
        self.log = log
        self.normalization = normalization

        if normalization == 'standard':
            if not (log and magnitude):
                warnings.warn(
                    'When `standard` passed to normalization method,\
                    both magnitude and log applied to the STFT. \
                    They\'re changed to proper setup'
                )
                self.log = True
                self.magnitude = True

            self.sclr = StandardScaler(*stft_standard_scaler_stats())
        elif normalization == 'sum2one':
            if log:
                warnings.warn(
                    'With `sum2one` normalization method, log should not be \
                    applied to the STFT. It\'s changed to proper setup.'
                )
                self.log = False

            if not magnitude:
                warnings.warn(
                    'With `sum2one` normalization method, magnitude should not be \
                    applied to the STFT. It\'s changed to proper setup.'
                )
                self.magnitude = True

            self.sclr = SumToOneNormalization(dim=1)
        else:  # None case
            self.sclr = Identity()
    
        self.stft = STFT(n_fft=n_fft, hop_sz=hop_sz,
                         magnitude=self.magnitude, log=self.log)
        test_x = self.stft(torch.randn(sig_len))
        self.input_shape = test_x.numpy().shape  # shape of input STFT

    def _preproc(self, x):
        return self.sclr(self.stft(x))[:, None]