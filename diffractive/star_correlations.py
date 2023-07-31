import argparse
import os

import scipy
import starfile
import torch
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pylab as plt
from tqdm import tqdm

import util.mrc
import diffractive_cpp
import math

from diffractive import relion
from diffractive.base import dt_symmetrize, integer_shift_2d, dft, euler_to_matrix, spectral_index_from_resolution, \
    grid_spectral_average, grid_spectral_sum
from diffractive.base.cache import Cache
from diffractive.base.explicit_grid_utils import size_to_maxr
from util.mrc import load_mrc
from util.mrc import save_mrc

from util.Euler import Euler_to_matrices, dmatrices_dEuler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('star', help='STAR file', type=str)
    parser.add_argument('label', type=str)
    parser.add_argument('--tabel', default="particles", type=str)

    args = parser.parse_args()

    star = starfile.read(args.star)
    data = star[args.tabel]

    x = np.array(data[args.label]).astype(float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)

    skipped = []

    for k in data:
        try:
            y = np.array(data[k]).astype(float)
            y = (y - np.mean(y)) / (np.std(y) + 1e-12)

            if np.std(y) == 0:
                raise ValueError()
            pearsonr = scipy.stats.pearsonr(x, y)
            print(f"{k}: \t {pearsonr.statistic}")
        except ValueError:
            skipped.append(k)

    print(f"Skipped labels {skipped}")








