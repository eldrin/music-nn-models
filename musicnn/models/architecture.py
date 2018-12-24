import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import STFT, StandardScaler, Identity
from ..datasets.files import stft_standard_scaler_stats


class BaseArchitecture(nn.Module):
    """Base Architecture for extension
    """
    def __init__(self):
        super().__init__()


class STFTInputNetwork(BaseArchitecture):
    """An architecture takes STFT as input representation

    Args:
        sig_len (int): input signal (1d) length
        n_fft (int): number of FFT points
        hop_sz (int): hop size for sliding windowing
        magnitude (bool): take the magnitude of the spectra
        log (bool): take the db-scale spectra
        z_score (bool): standardizing spectra
    """
    def __init__(self, sig_len, n_fft, hop_sz, magnitude=True,
                 log=True, z_score=False):
        super().__init__()
        self.sig_len = sig_len
        self.n_fft = n_fft
        self.hop_sz = hop_sz
        self.magnitude = magnitude
        self.log = log
        self.z_score = z_score

        if z_score:
            self.sclr = StandardScaler(*stft_standard_scaler_stats())
        else:
            self.sclr = Identity()
    
        self.stft = STFT(n_fft=n_fft, hop_sz=hop_sz,
                         magnitude=magnitude, log=log)
        test_x = self.stft(torch.randn(sig_len))
        self.input_shape = test_x.numpy().shape  # shape of input STFT

    def _preproc(self, x):
        return self.sclr(self.stft(x))[:, None]