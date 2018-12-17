import numpy as np
import librosa

import torch
from torch.nn import functional as F


def mu_law_encode(audio, quantization_channels, is_gpu=False):
    """ Quantizes waveform amplitudes.
    PyTorch implementation of
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64
    """
    audio = torch.tensor(audio).float()
    mu = torch.tensor(quantization_channels - 1.)
    if is_gpu:
        audio = audio.cuda()
        mu = mu.cuda()

    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = torch.min(audio.abs(), torch.tensor([1.]))
    magnitude = torch.log1p(mu * safe_audio_abs) / torch.log1p(mu)
    signal = torch.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).int()


def mu_law_decode(output, quantization_channels, is_gpu=False):
    """ Recovers waveform from quantized values.
    PyTorch implementation of
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L78
    """
    output = torch.tensor(output)
    mu = torch.tensor(quantization_channels - 1.)
    if is_gpu:
        output = output.cuda()
        mu = mu.cuda()

    # Map values back to [-1, 1].
    signal = 2 * (output.float() / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**signal.abs() - 1)
    return torch.sign(signal) * magnitude