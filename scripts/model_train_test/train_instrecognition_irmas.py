import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial

from torchvision.transforms import Compose

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets.instrecognition import CLS
from musicnn.datasets import IRMASTraining
from musicnn.trainers import InstRecognitionTrainer
from musicnn.models import VGGlike2DAutoTagger


# setup variables
audio_root = '/data/IRMAS/IRMAS_TRN_NPY/'
model_path = '/data/models/IRMAS_test'
fold = 0
n_tags = len(CLS)

# load the dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
IRMASTraining_ = partial(
    IRMASTraining, songs_root=audio_root, fold=fold, transform=transformer
)
train_dataset = IRMASTraining_(split='train')
valid_dataset = IRMASTraining_(split='valid')

# spawn a model
model = VGGlike2DAutoTagger(n_tags)

# spawn a trainer
trainer = InstRecognitionTrainer(
    model         = model,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 128,
    n_epochs      = 1000,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 20,
    save_every    = 50
)

# run
trainer.fit()