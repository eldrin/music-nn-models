import os
from os.path import join, dirname, basename
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))
import argparse

import librosa
import random
import numpy as np
import musdb
from tqdm import tqdm

from musicnn.utils.ops import mu_law_encode
from musicnn.config import Config as cfg


IX_SRC_MAP = {
    'mixture': 0,
    'drums': 1,
    'bass': 2,
    'accompaniment': 3,
    'vocals': 4
}


def _process(track, targets=['mixture', 'vocals']):
    """inner processing block (retrieval, resampling, mulaw)

    Args:
        track (musdb.Track): track object containing stems and metadata
    
    Returns:
        dict: contains source name (key) and signal (numpy.ndarray)
    """
    sr = track.rate
    stems = track.stems
    output = {}
    for src_name in targets:
        # retrieval
        signal = stems[IX_SRC_MAP[src_name]]
        # make mono
        signal = signal.mean(-1)
        # resampling
        signal = librosa.resample(
            signal, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
        # mu-law
        signal = mu_law_encode(signal, cfg.QUANTIZATION_CHANNELS).numpy()
        # register
        output[src_name] = signal.astype(np.uint8)
    
    return output


def process(musdb_root, output_root, valid_ratio=0.1):
    """Main processing loop. conduct preprocessing and dump

    Args:
        musdb_root (str): path to the MUSDB18 dataset
        output_root (str): path to dump processed data (.npy)
        valid_ratio (float): validation set ratio
    """
    mus = musdb.DB(root_dir=musdb_root)
    out_paths = {
        'train': join(output_root, 'train'),
        'valid': join(output_root, 'valid'),
        'test': join(output_root, 'test')
    }

    # parsing train / valid set from original training set
    train_tracks = mus.load_mus_tracks(subsets=['train'])
    random.shuffle(train_tracks)
    n_trains = int(len(train_tracks) * (1 - valid_ratio))
    train_tracks, valid_tracks = train_tracks[:n_trains], train_tracks[n_trains:]
    # get test tracks
    test_tracks = mus.load_mus_tracks(subsets=['test'])

    # main process
    for split, tracks in zip(['train', 'valid', 'test'],
                             [train_tracks, valid_tracks, test_tracks]):

        print('Processing {} set...'.format(split))
        if not os.path.exists(out_paths[split]):
            os.mkdir(out_paths[split])

        for track in tqdm(tracks, ncols=80):
            for src_name, signal in _process(track).items():
                np.save(
                    join(
                        out_paths[split],
                        '{}_{}.npy'.format(track.name, src_name)
                    ),
                    signal
                )


if __name__ == "__main__":
    # setup arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("musdb_root", help='path to MUSDB18 is stored')
    parser.add_argument("output_root", help='path to processed data is stored')
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="ratio for the validation set")
    args = parser.parse_args()

    process(args.musdb_root, args.output_root, args.valid_ratio)