import os
from os.path import join, dirname, basename
import sys
# add repo root to the system path
sys.path.append(join(dirname(__file__), '../..'))

import librosa
import numpy as np
import musdb

from musicnn.utils.ops import mu_law_encode
from musicnn.config import Config as cfg

