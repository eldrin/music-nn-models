from .autotagging import VGGlike2DAutoTagger, ShallowAutoTagger, MFCCAutoTagger
from .autoencoder import VGGlike2DAutoEncoder
from .sourceseparation import VGGlike2DUNet

__all__ = ['VGGlike2DAutoTagger',
           'VGGlike2DAutoEncoder',
           'VGGlike2DUNet',
           'ShallowAutoTagger',
           'MFCCAutoTagger']
