import torch
import util.mrc
import diffractive_cpp
import pylab as plt
import math

from util.mrc import load_mrc
from util.mrc import save_mrc

from util.Euler import Euler_to_matrices, dmatrices_dEuler


image = load_mrc('test.mrc')[0]

v = torch.from_numpy(image)
V = torch.fft.rfftn(v)
diffractive_cpp.shift_center_3d(V);

b = 12


a = torch.arange(b) * 2 * math.pi / b

s = torch.sin(a)
c = torch.cos(a)
z = torch.zeros(b)
o = torch.ones(b)

r1 = torch.stack([c,  z, -s], 1)
r2 = torch.stack([z,  o,  z], 1)
r3 = torch.stack([s,  z,  c], 1)

R  = torch.stack([r1, r2, r3], 1)


proj_FT = diffractive_cpp.complex_forward(V, R);

proj_FT = proj_FT.cpu()

diffractive_cpp.shift_center_stack(proj_FT);

proj = torch.fft.irfft2(proj_FT)

save_mrc(proj.detach().numpy(), 1, 'test_betagal_rotY_cpu.mrc')



