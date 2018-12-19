import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets.autotagging import MSDLastFM50, TAGS
from musicnn.trainers.autotagging import AutoTaggingTrainer
from musicnn.models.autotagging import VGGlike2DAutoTagger


# setup variables
audio_root = '/home/jaykim/Documents/datasets/MSD/npy/'
model_path = '/data/models/MSDLastFM50_Test_dropout'
fold = 0
n_tags = len(TAGS)

# load the dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
MSDLastFM50_ = partial(
    MSDLastFM50, songs_root=audio_root, fold=fold, transform=transformer
)
train_dataset = MSDLastFM50_(split='train')
valid_dataset = MSDLastFM50_(split='valid')

# spawn a model
model = VGGlike2DAutoTagger(n_tags)

# spawn a trainer
trainer = AutoTaggingTrainer(
    model         = model,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 128,
    n_epochs      = 5000,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 20,
    save_every    = 50
)

# run
trainer.fit()