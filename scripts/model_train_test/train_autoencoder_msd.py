import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets import MSDAudio
from musicnn.trainers import AutoEncoderTrainer
from musicnn.models import VGGlike2DAutoEncoder


# setup variables
audio_root = '/home/jaykim/Documents/datasets/MSD/npy/'
model_path = '/data/models/MSDAE_kl_test'
fold = 0

# load the dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
MSDAudio_ = partial(
    MSDAudio, songs_root=audio_root, fold=fold, transform=transformer
)
train_dataset = MSDAudio_(split='train')
valid_dataset = MSDAudio_(split='valid')

# spawn a model
model = VGGlike2DAutoEncoder(normalization='sum2one')

# spawn a trainer
trainer = AutoEncoderTrainer(
    model         = model,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 64,
    n_epochs      = 5000,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 20,
    save_every    = 50
)

# run
trainer.fit()