import torch
import util.mrc
import diffractive_cpp
import pylab as plt
import math
import projections

from util.mrc import load_mrc
from util.mrc import save_mrc
from util.Euler import Euler_to_matrices

s = 16
real_image = torch.rand((s,s,s))

v = real_image.double()
V = torch.fft.rfftn(v)
diffractive_cpp.shift_center_3d(V);

b = 1

a = torch.rand((1,3)) * 2 * math.pi / b

R = Euler_to_matrices(a)
R = R.double()
R.requires_grad = True

V = V.cuda()
R = R.cuda()

torch.autograd.gradcheck(
	projections.ComplexMatrixProjection.apply, 
	(V, R), eps=1e-6, atol=1e-4)
