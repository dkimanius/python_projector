import torch
import numpy as np
import math


def Euler_to_matrices(angles):
	
	alpha = angles[:,0]
	beta  = angles[:,1]
	gamma = angles[:,2]
	
	ca = torch.cos(alpha)
	cb = torch.cos(beta)
	cg = torch.cos(gamma)
	sa = torch.sin(alpha)
	sb = torch.sin(beta)
	sg = torch.sin(gamma)
	
	cc = cb * ca
	cs = cb * sa
	sc = sb * ca
	ss = sb * sa
	
	a1 = torch.stack([ cg * cc - sg * sa,   cg * cs + sg * ca,  -cg * sb], 1)
	a2 = torch.stack([-sg * cc - cg * sa,  -sg * cs + cg * ca,   sg * sb], 1)
	a3 = torch.stack([                sc,                  ss,        cb], 1)
	
	A  = torch.stack([a1, a2, a3], 1)	
	
	return A
	
	
def dmatrices_dEuler(angles):
	
	b = angles.size(0)
	
	alpha = angles[:,0]
	beta  = angles[:,1]
	gamma = angles[:,2]
	
	ca = torch.cos(alpha)
	cb = torch.cos(beta)
	cg = torch.cos(gamma)
	sa = torch.sin(alpha)
	sb = torch.sin(beta)
	sg = torch.sin(gamma)
	
	dca = -torch.sin(alpha)
	dcb = -torch.sin(beta)
	dcg = -torch.sin(gamma)
	dsa = torch.cos(alpha)
	dsb = torch.cos(beta)
	dsg = torch.cos(gamma)
	
	z = torch.zeros(b, device=angles.device)
	
	a1a = torch.stack([ cg * cb * dca - sg * dsa,   cg * cb * dsa + sg * dca,  z], 1)
	a2a = torch.stack([-sg * cb * dca - cg * dsa,  -sg * cb * dsa + cg * dca,  z], 1)
	a3a = torch.stack([                 sb * dca,                   sb * dsa,  z], 1)
	
	Aa  = torch.stack([a1a, a2a, a3a], 1)
	
	a1b = torch.stack([ cg * dcb * ca,    cg * dcb * sa,  -cg * dsb], 1)
	a2b = torch.stack([-sg * dcb * ca,   -sg * dcb * sa,   sg * dsb], 1)
	a3b = torch.stack([      dsb * ca,         dsb * sa,        dcb], 1)
	
	Ab  = torch.stack([a1b, a2b, a3b], 1)
	
	a1g = torch.stack([ dcg * cb * ca - dsg * sa,   dcg * cb * sa + dsg * ca, -dcg * sb], 1)
	a2g = torch.stack([-dsg * cb * ca - dcg * sa,  -dsg * cb * sa + dcg * ca,  dsg * sb], 1)
	a3g = torch.stack([                        z,                          z,         z], 1)
	
	Ag  = torch.stack([a1g, a2g, a3g], 1)
	
	Aabg  = torch.stack([Aa, Ab, Ag], 1)
	
	return Aabg

