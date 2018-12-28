import pkg_resources
import pickle as pkl

import numpy as np
import pandas as pd

from .audiodataset import AudioDataset
from .files import msd_lastfm50_label, msd_lastfm50_splits

# from: https://github.com/keunwoochoi/MSD_split_for_tagging
TAGS = ['rock', 'pop', 'alternative', 'indie', 'electronic',
        'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
        'beautiful', 'metal', 'chillout', 'male vocalists',
        'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
        '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
        'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
        'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
        '70s', 'party', 'country', 'easy listening',
        'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
        'Progressive rock', '60s', 'rnb', 'indie pop',
        'sad', 'House', 'happy']
TAGS = {tag: ix for ix, tag in enumerate(TAGS)}


class MSDLastFM50(AudioDataset):
    """MSD-LastFM top 50 tag dataset

    Tag selection is based on Keunwoo Choi's work

    .._Keunwoo's git
    https://github.com/keunwoochoi/MSD_split_for_tagging

    """
    def __init__(self, songs_root, split='train', fold=0,
                 crop_len=44100, transform=None, on_mem=False):
        """"""
        # load target
        track_tag_map = pkl.load(open(msd_lastfm50_label(), 'rb'))

        # load split info
        split_info = msd_lastfm50_splits(fold, split)

        # call super class' constructor
        super().__init__(songs_root, split_info,
                         target=track_tag_map, crop_len=crop_len,
                         transform=transform, on_mem=on_mem)

    def _retrieve_target(self, fn):
        """"""
        ix = np.array([TAGS[tag] for tag in self.target[fn]])
        out = np.zeros((len(TAGS),), dtype=np.float32)
        out[ix] = 1
        return out