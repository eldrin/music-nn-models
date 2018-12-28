from os.path import dirname
import pkg_resources
import pickle as pkl

import numpy as np
import pandas as pd

from .audiodataset import AudioDataset
from .files import irmas_train_splits 

CLS = ['cel', 'cla', 'flu', 'gac', 'gel', 'org',
       'pia', 'sax', 'tru', 'vio', 'voi']
CLS = {tag: ix for ix, tag in enumerate(CLS)}


class IRMASTraining(AudioDataset):
    """Training set of IRMAS dataset

    The training dataset of IRMAS can be formulated as
    single-label multi-class classification. It has in total 11 classes,
    which is saved in the global variable CLS.

    .._IRMAS dataset
    https://www.upf.edu/web/mtg/irmas
    """
    def __init__(self, songs_root, split='train', fold=0,
                 crop_len=44100, transform=None, on_mem=True):
        """"""
        # load split info
        split_info = irmas_train_splits(fold, split)

        # call super class' constructor
        super().__init__(songs_root, split_info,
                         target=None, crop_len=crop_len,
                         transform=transform, on_mem=True)

        # build target map
        self.target = {fn: dirname(fn) for fn in self.subset_fns}

    def _retrieve_target(self, fn):
        """"""
        return CLS[self.target[fn]]