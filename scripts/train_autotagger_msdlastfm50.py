import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets.autotagging import MSDLastFM50
from musicnn.trainers.autotagging import AutoTaggingTrainer
from musicnn.models.autotagging import VGGlike2DAutoTagger


# setup variables
audio_root = '/home/jaykim/Documents/datasets/MSD/npy/'
fold = 0

# load the dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
MSDLastFM50_ = partial(MSDLastFM50, song_root=audio_root, fold=fold)
train_dataset = MSDLastFM50_(split='train')
valid_datsaet = MSDLastFM50_(split='valid')

# spawn a model
# model = VGGlike2DAutoTagger()