import os
from os.path import join, basename, dirname
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
from functools import partial
import pickle as pkl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from musicnn.models import ShallowAutoTagger
from musicnn.trainers import ShallowAutoTaggingTrainer
from musicnn.datasets import files
from musicnn.datasets.autotagging import TAGS


def convert_tags_to_onehot(tags):
    y = np.zeros((len(TAGS),))
    for tag in tags:
        y[TAGS[tag]] = 1
    return y


MSD_LASTFM50_LABEL = pkl.load(open(files.msd_lastfm50_label(), 'rb'))
def get_labels(fn_list_fn):
    labels = []
    with open(fn_list_fn) as f:
        for line in f.readlines():
            audio_id = basename(line.replace('\n', ''))
            labels.append(
                convert_tags_to_onehot(
                    MSD_LASTFM50_LABEL[audio_id]
                )
            )
    return np.array(labels)  # (n_files, n_tags)

# setup variables
data_root = '/mnt/data/nmd/'
model_path = '/mnt/data/nmd/models/mfcc_at_fat_nodropout_trial4'
dropout = 0

# load the dataset
# get features
Xtr = torch.FloatTensor(np.load(join(data_root, 'msdlastfm50_train_0_mfcc.npy')))
Xvl = torch.FloatTensor(np.load(join(data_root, 'msdlastfm50_valid_0_mfcc.npy')))
Xts = torch.FloatTensor(np.load(join(data_root, 'msdlastfm50_test_0_mfcc.npy')))

# get labels
label_fn = partial(files.msd_lastfm50_splits, fold=0)
ytr = torch.FloatTensor(get_labels(label_fn(split='train')))
yvl = torch.FloatTensor(get_labels(label_fn(split='valid')))
yts = torch.FloatTensor(get_labels(label_fn(split='test')))

# sampling 70%
rnd_idx = np.random.permutation(Xtr.shape[0])
n_samples = int(len(rnd_idx) * 0.7)
Xtr = Xtr[rnd_idx[:n_samples]]
ytr = ytr[rnd_idx[:n_samples]]

# setup dataset & dataloader
train = TensorDataset(Xtr, ytr)
valid = TensorDataset(Xvl, yvl)
test = TensorDataset(Xts, yts)

# prepare training
model = ShallowAutoTagger(len(TAGS), Xtr.shape[-1], dropout=dropout)
trainer = ShallowAutoTaggingTrainer(
    model         = model,
    train_dataset = train,
    valid_dataset = valid,
    l2            = 1e-7,
    learn_rate    = 0.001,
    batch_size    = 24,
    n_epochs      = 500,
    is_gpu        = True,
    checkpoint    = model_path,
    loss_every    = 100,
    save_every    = 100
)

# run
trainer.fit()
