import torch
import diffractive_cpp
import numpy as np
from util.Euler import Euler_to_matrices, dmatrices_dEuler
import math
	

class ComplexProjection(torch.autograd.Function):
	
	@staticmethod
	def forward(context, map, angles):
		R = Euler_to_matrices(angles)
		proj = diffractive_cpp.complex_forward(map, R)
		context.save_for_backward(map, angles, R)
		return proj
	
	@staticmethod
	def backward(context, grad_proj):
		map, angles, R = context.saved_tensors
		grad_map, grad_R = diffractive_cpp.complex_backward(map, R, grad_proj)
		dR_dangles = dmatrices_dEuler(angles)
		grad_angle_0 = (grad_R * dR_dangles[:,0,:,:]).sum((1,2))
		grad_angle_1 = (grad_R * dR_dangles[:,1,:,:]).sum((1,2))
		grad_angle_2 = (grad_R * dR_dangles[:,2,:,:]).sum((1,2))
		grad_angles = torch.stack([grad_angle_0, grad_angle_1, grad_angle_2], 1)
		return grad_map, grad_angles


class ComplexMatrixProjection(torch.autograd.Function):
	
	@staticmethod
	def forward(context, map, R):
		proj = diffractive_cpp.complex_forward(map, R)
		context.save_for_backward(map, R)
		return proj
	
	@staticmethod
	def backward(context, grad_proj):
		map, R = context.saved_tensors
		grad_map, grad_R = diffractive_cpp.complex_backward(map, R, grad_proj)
		return grad_map, grad_R


class RealProjection(torch.autograd.Function):
	
	@staticmethod
	def forward(context, map, angles):
		R = Euler_to_matrices(angles)
		proj = diffractive_cpp.real_forward(map, R)
		context.save_for_backward(map, angles, R)
		return proj
	
	@staticmethod
	def backward(context, grad_proj):
		map, angles, R = context.saved_tensors
		grad_map, grad_R = diffractive_cpp.real_backward(map, R, grad_proj)
		dR_dangles = dmatrices_dEuler(angles)
		grad_angle_0 = (grad_R * dR_dangles[:,0,:,:]).sum((1,2))
		grad_angle_1 = (grad_R * dR_dangles[:,1,:,:]).sum((1,2))
		grad_angle_2 = (grad_R * dR_dangles[:,2,:,:]).sum((1,2))
		grad_angles = torch.stack([grad_angle_0, grad_angle_1, grad_angle_2], 1)
		return grad_map, grad_angles
		

class RealMatrixProjection(torch.autograd.Function):
	
	@staticmethod
	def forward(context, map, R):
		proj = diffractive_cpp.real_forward(map, R)
		context.save_for_backward(map, R)
		return proj
	
	@staticmethod
	def backward(context, grad_proj):
		map, R = context.saved_tensors
		grad_map, grad_R = diffractive_cpp.real_backward(map, R, grad_proj)
		return grad_map, grad_R
