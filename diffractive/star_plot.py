import argparse
import os

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
    parser.add_argument('xlabel', type=str)
    parser.add_argument('ylabel', type=str)
    parser.add_argument('--tabel', default="particles", type=str)

    args = parser.parse_args()

    star = starfile.read(args.star)

    x = np.array(star[args.tabel][args.xlabel]).astype(float)
    y = np.array(star[args.tabel][args.ylabel]).astype(float)

    plt.scatter(x, y, marker='.', alpha=0.1, edgecolors="none")
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.show()








