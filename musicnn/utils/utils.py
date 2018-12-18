from os.path import basename, join, dirname
import shutil
from multiprocessing import Pool

from scipy import signal as sig
import numpy as np

import torch
from tqdm import tqdm


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """"""
    torch.save(state, filename)
    if is_best:
        new_basename = basename(filename)
        new_basename = new_basename.replace('checkpoint', 'model_best')
        new_fn = join(dirname(filename), new_basename)
        shutil.copyfile(filename, new_fn)


def load_checkpoint(filename):
    """"""
    checkpoint = torch.load(
        filename, map_location=lambda storage, loc: storage)
    return checkpoint


def parmap(func, iterable, n_workers=2, verbose=False):
    """ Simple Implementation for Parallel Map """

    if n_workers == 1:
        if verbose:
            iterable = tqdm(iterable, total=len(iterable), ncols=80)
        return map(func, iterable)
    else:
        with Pool(processes=n_workers) as p:
            if verbose:
                with tqdm(total=len(iterable), ncols=80) as pbar:
                    output = []
                    for o in p.imap_unordered(func, iterable):
                        output.append(o)
                        pbar.update()
                return output
            else:
                return p.imap_unordered(func, iterable)