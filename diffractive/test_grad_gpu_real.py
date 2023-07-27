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

V = real_image.double()

b = 1

a = torch.rand((1,3)) * 2 * math.pi / b

R = Euler_to_matrices(a)
a = a.double()
a.requires_grad = True

V = V.cuda()
a = a.cuda()

torch.autograd.gradcheck(
	projections.RealProjection.apply, 
	(V, a), eps=1e-6, atol=1e-4)
