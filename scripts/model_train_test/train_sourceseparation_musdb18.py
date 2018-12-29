import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets import VocalSeparation
from musicnn.trainers import SourceSeparationTrainer
from musicnn.models import VGGlike2DUNet


# setup variables
audio_root = '/tudelft.net/staff-bulk/ewi/insy/MMC/jaykim/datasets/musdb18npy/'
model_path = '/tudelft.net/staff-bulk/ewi/insy/MMC/jaykim/datasets/musdb18npy/models/SS_test'
fold = 0

# load the dataset
transformer = Compose([
    MuLawDecoding(
        cfg.QUANTIZATION_CHANNELS,
        target_keys=['mixture', 'vocals']
    )
])
VocalSeparation_ = partial(
    VocalSeparation, songs_root=audio_root, fold=fold, transform=transformer
)
train_dataset = VocalSeparation_(split='train')
valid_dataset = VocalSeparation_(split='valid')

# spawn a model
model = VGGlike2DUNet()

# spawn a trainer
trainer = SourceSeparationTrainer(
    model         = model,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 48,
    n_epochs      = 5000,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 20,
    save_every    = 50
)

# run
trainer.fit()
