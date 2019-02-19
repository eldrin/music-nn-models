from .autotagging import AutoTaggingTrainer, ShallowAutoTaggingTrainer
from .autoencoder import AutoEncoderTrainer
from .instrecognition import InstRecognitionTrainer
from .sourceseparation import SourceSeparationTrainer

__all__ = [
    'AutoTaggingTrainer',
    'ShallowAutoTaggingTrainer',
    'AutoEncoderTrainer',
    'InstRecognitionTrainer',
    'SourceSeparationTrainer'
]
