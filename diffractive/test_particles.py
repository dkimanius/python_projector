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


def get_spectrum_from_grid_(grid, sum=True):
    mask = Cache.get_spectral_mask(
        grid.shape[-2:],
        max_r=size_to_maxr(grid.shape[-2]),
        device=grid.device
    )
    indices = Cache.get_spectral_indices(
        grid.shape[-2:],
        max_r=size_to_maxr(grid.shape[-2]),
        device=grid.device
    )[mask]

    grid = grid[mask] if len(grid.shape) == 2 else grid[:, mask]
    if sum:
        return grid_spectral_sum(grid, indices)
    else:
        return grid_spectral_average(grid, indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--half1', help='MRC file', type=str)
    parser.add_argument('--half2', help='MRC file', type=str)
    parser.add_argument('--star', help='STAR file', type=str)
    parser.add_argument('--minres', help='Minium resolution to consider', type=float, default=None)
    parser.add_argument('--maxres', help='Maximum resolution to consider', type=float, default=None)
    parser.add_argument('--circular-mask-thickness', help='Circular mask thickness', type=float, default=5)
    parser.add_argument('--particle-diameter', help='Circular mask diameter', type=float, default=None)
    parser.add_argument('--plot', help='Plot each particle', action='store_true')
    parser.add_argument('--threads', help='threads for loading data', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=50)

    args = parser.parse_args()

    device = "cuda:0"

    relion_dataset = relion.RelionDataset(
        args.star,
        dtype=np.float32,
    )
    dataset = relion_dataset.make_particle_dataset()
    star = relion_dataset.get_starfile()

    image_size = dataset.get_output_image_size(0)
    pixel_size = dataset.get_output_pixel_size(0)
    max_r = size_to_maxr(image_size)

    max_diameter_ang = image_size * pixel_size - args.circular_mask_thickness

    if args.maxres is not None:
        min_spectral_index = spectral_index_from_resolution(
            args.maxres, image_size=image_size, pixel_size=pixel_size)
    else:
        min_spectral_index = 0

    if args.minres is not None:
        max_spectral_index = spectral_index_from_resolution(
            args.minres, image_size=image_size, pixel_size=pixel_size)
    else:
        max_spectral_index = max_r

    spectral_mask = torch.ones(max_r).bool().to(device)
    spectral_mask[:min_spectral_index] = False
    spectral_mask[max_spectral_index:] = False

    if args.particle_diameter is None:
        diameter_ang = image_size * 0.75 * pixel_size - args.circular_mask_thickness
        print(f"Assigning a diameter of {round(diameter_ang)} angstrom")
    else:
        if args.particle_diameter > max_diameter_ang:
            print(
                f"WARNING: Specified particle diameter {round(args.particle_diameter)} angstrom is too large\n"
                f" Assigning a diameter of {round(max_diameter_ang)} angstrom"
            )
            diameter_ang = max_diameter_ang
        else:
            diameter_ang = args.particle_diameter

    circular_mask_thickness = args.circular_mask_thickness
    circular_mask_radius = diameter_ang / (2 * pixel_size)

    v1 = load_mrc(args.half1)[0].copy()
    v1 = torch.from_numpy(v1)
    V1 = torch.fft.rfftn(v1)
    diffractive_cpp.shift_center_3d(V1)
    V1 = V1.to(device)

    v2 = load_mrc(args.half2)[0].copy()
    v2 = torch.from_numpy(v2)
    V2 = torch.fft.rfftn(v2)
    diffractive_cpp.shift_center_3d(V2)
    V2 = V2.to(device)

    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=args.threads,
        batch_size=args.batch_size
    )

    torch.no_grad()

    star['particles']['MinFrcMean'] = 0
    star['particles']['MaskedMinFrcMean'] = 0

    nr_parts = len(data_loader)
    pbar = tqdm(total=nr_parts, smoothing=0.1)

    for sample in data_loader:
        y = sample['image'].to(device)
        image_size = y.shape[-1]
        group_idx = sample['noise_group_idx'].to(device)
        y = Cache.apply_square_mask(y, thickness=circular_mask_thickness)

        ctf = dt_symmetrize(sample["ctf"], dim=2)[:, :-1, image_size // 2:].to(device)
        shifts = sample["translation"]
        shifts_int = torch.round(shifts).long().detach()
        shifts_resid = shifts - shifts_int

        y_ = integer_shift_2d(y, shifts_int)
        y_ = Cache.apply_circular_mask(y_, thickness=circular_mask_thickness, radius=circular_mask_radius)
        y = dft(y_, dim=2, real_in=True)

        R = euler_to_matrix(sample['rotation']).to(device)

        x1 = diffractive_cpp.complex_forward(V1, R)
        x2 = diffractive_cpp.complex_forward(V2, R)

        x1 = torch.fft.fftshift(x1, axis=1)
        x2 = torch.fft.fftshift(x2, axis=1)

        x1 *= ctf
        x2 *= ctf

        cc1 = y.real * x1.real + y.imag * x1.imag
        cc2 = y.real * x2.real + y.imag * x2.imag

        cc1_spec = get_spectrum_from_grid_(cc1, sum=True)
        cc2_spec = get_spectrum_from_grid_(cc2, sum=True)

        x1_pow = get_spectrum_from_grid_(x1.abs().square(), sum=True)
        x2_pow = get_spectrum_from_grid_(x2.abs().square(), sum=True)
        y_pow = get_spectrum_from_grid_(y.abs().square(), sum=True)

        frc1 = cc1_spec / (torch.sqrt(x1_pow * y_pow) + 1e-12)
        frc2 = cc2_spec / (torch.sqrt(x2_pow * y_pow) + 1e-12)

        if args.plot:
            plt.plot(np.arange(max_r), frc1[0].detach().cpu().numpy())
            plt.plot(np.arange(max_r), frc2[0].detach().cpu().numpy())
            plt.show()

        frc1_mean = torch.mean(frc1, dim=1)
        frc2_mean = torch.mean(frc2, dim=1)

        frc1_mask_mean = torch.mean(frc1[:, spectral_mask], dim=1)
        frc2_mask_mean = torch.mean(frc2[:, spectral_mask], dim=1)

        MinFrcMean = torch.minimum(frc1_mean, frc2_mean).cpu().detach().numpy()
        MaskedMinFrcMean = torch.minimum(frc1_mask_mean, frc2_mask_mean).cpu().detach().numpy()

        particle_idx = sample["idx"].cpu().detach().numpy()
        for i, idx in enumerate(particle_idx):
            star['particles'].at[idx, 'MinFrcMean'] = MinFrcMean[i]
            star['particles'].at[idx, 'MaskedMinFrcMean'] = MaskedMinFrcMean[i]
            pbar.update()

    pbar.close()
    output_fn = os.path.splitext(args.star)[0]
    output_fn = f"{output_fn}_frc.star"
    starfile.write(star, output_fn, overwrite=True)

    print(f"Saved to {output_fn}")








