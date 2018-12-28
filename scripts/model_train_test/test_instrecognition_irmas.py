import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score

import torch
from torchvision.transforms import Compose

from tqdm import trange

from musicnn.config import Config as cfg
from musicnn.datasets.audiodataset import MuLawDecoding
from musicnn.datasets.instrecognition import CLS
from musicnn.datasets import IRMASTraining
from musicnn.models import VGGlike2DAutoTagger

# load the checkpoint
checkpoint = torch.load(
    '/data/models/MSDLastFM50_Test_dropout_sclr_it150.pth',
    lambda a, b: a  # make sure the model is loaded on CPU
)

# initialize model and load the checkpoint
model = VGGlike2DAutoTagger(len(CLS))
model.eval()
model.load_state_dict(checkpoint['state_dict'])

# get the test dataset
transformer = Compose([MuLawDecoding(cfg.QUANTIZATION_CHANNELS)])
test_dataset = IRMASTraining(
    songs_root = '/home/jaykim/Documents/datasets/MSD/npy/',
    fold       = 0,
    transform  = transformer,
    split      = 'test'
)

# get the outputs
batch_size = 64
TRUE, PRED = [], []
for start in trange(0, len(test_dataset), batch_size, ncols=80):

    end = (
        start + batch_size
        if start + batch_size <= len(test_dataset)
        else len(test_dataset)
    )

    samples = [test_dataset[j] for j in range(start, end)]
    x = np.array([sample['signal'] for sample in samples])
    x = torch.from_numpy(x)

    TRUE.extend([sample['target'] for sample in samples])
    PRED.append(torch.argmax(model(x), dim=-1).data.numpy())

TRUE, PRED = np.array(TRUE), np.concatenate(PRED, axis=0)

# claculate metrics
print('ACC:', accuracy_score(TRUE, PRED))
