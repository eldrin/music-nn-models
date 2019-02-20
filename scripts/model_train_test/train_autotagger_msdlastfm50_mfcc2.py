import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets.autotagging import TAGS
from musicnn.datasets import MSDLastFM50
from musicnn.trainers import AutoTaggingTrainer
from musicnn.models import MFCCAutoTagger


# setup variables
# audio_root = '/home/jaykim/Documents/datasets/MSD/npy/'
# model_path = '/data/models/MSDLastFM50_mfcc_test'
audio_root = '/mnt/data/msd/'
model_path = '/mnt/data/nmd_data/models/MSDLastFM50_mfcc_test'
fold = 0
n_tags = len(TAGS)

# load the dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
MSDLastFM50_ = partial(
    MSDLastFM50, songs_root=audio_root, fold=fold, transform=transformer
)
train_dataset = MSDLastFM50_(split='train', sampling=0.7)
valid_dataset = MSDLastFM50_(split='valid')

# spawn a model
model = MFCCAutoTagger(n_tags)

# spawn a trainer
trainer = AutoTaggingTrainer(
    model         = model,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 24,
    n_epochs      = 500,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 20,
    save_every    = 100
)

# run
trainer.fit()
