import numpy as np
import mrcfile as mrc
import torch
from scipy import interpolate
from typing import Tuple, Union, TypeVar

TensorType = TypeVar('torch.tensor')


def save_mrc(grid, voxel_size, filename):
	
	if grid.ndim == 2:
		(y, x) = grid.shape
		z = 1
	else:
		(z, y, x) = grid.shape
		
	o = mrc.new(filename, overwrite=True)
	
	o.header['cella'].x = x * voxel_size
	o.header['cella'].y = y * voxel_size
	o.header['cella'].z = z * voxel_size
	o.header['cellb'].x = x * voxel_size
	o.header['cellb'].y = y * voxel_size
	o.header['cellb'].z = z * voxel_size
	o.header['origin'].x = 0
	o.header['origin'].y = 0
	o.header['origin'].z = 0
	
	out_box = np.reshape(grid, (z, y, x))
	o.set_data(out_box.astype(np.float32))
	o.update_header_stats()
	o.flush()
	o.close()


def load_mrc(mrc_fn):
	mrc_file = mrc.open(mrc_fn, 'r')
	c = mrc_file.header['mapc']
	r = mrc_file.header['mapr']
	s = mrc_file.header['maps']

	global_origin = mrc_file.header['origin']
	global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
	global_origin[0] += mrc_file.header['nxstart']
	global_origin[1] += mrc_file.header['nystart']
	global_origin[2] += mrc_file.header['nzstart']

	global_origin *= mrc_file.voxel_size.x

	if c == 1 and r == 2 and s == 3:
		grid = mrc_file.data
	elif c == 3 and r == 2 and s == 1:
		grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
	elif c == 2 and r == 1 and s == 3:
		grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
	else:
		raise RuntimeError("MRC file axis arrangement not supported!")

	return grid, mrc_file.voxel_size.x, global_origin

