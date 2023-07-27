#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

#include "cpu_checks.h"

template <typename scalar_t>
void shift_center_3d_cpu_kernel(
	torch::TensorAccessor<scalar_t, 3> acc,
	int w, int h, int d)
{
	for (int z = 0; z < d; z++)
	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++)
	{
		const float sign = 
			(1 - 2 * (x % 2)) * 
			(1 - 2 * (y % 2)) * 
			(1 - 2 * (z % 2)) ;
		
		acc[z][y][x] *= sign;
	}
}

template <typename scalar_t>
void shift_center_stack_cpu_kernel(
	torch::TensorAccessor<scalar_t, 3> acc,
	int w, int h, int d)
{
	for (int z = 0; z < d; z++)
	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++)
	{
		const float sign = 
			(1 - 2 * (x % 2)) * 
			(1 - 2 * (y % 2)) ;
		
		acc[z][y][x] *= sign;
	}
}

void shift_center_3d_cpu(
	torch::Tensor image)
{
	CHECK_INPUT(image)
	
	AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
		image.scalar_type(),
		"trilinear_complex_projection_forward_cpu_kernel",
		([&] {
				shift_center_3d_cpu_kernel<scalar_t>(
					image.accessor<scalar_t, 3>(),
					image.size(2),
					image.size(1),
					image.size(0)
				);
		})
	);	
}

void shift_center_stack_cpu(
	torch::Tensor image)
{
	CHECK_INPUT(image)
	
	AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
		image.scalar_type(),
		"trilinear_complex_projection_forward_cpu_kernel",
		([&] {
				shift_center_stack_cpu_kernel<scalar_t>(
					image.accessor<scalar_t, 3>(),
					image.size(2),
					image.size(1),
					image.size(0)
				);
		})
	);	
}

void shift_center_3d(
	torch::Tensor image)
{
	if (image.device().type() == torch::kCPU)
	{
		shift_center_3d_cpu(image);
	}
	else
	{
		throw std::logic_error("Support for device not implemented");
	}
}

void shift_center_stack(
	torch::Tensor image)
{
	if (image.device().type() == torch::kCPU)
	{
		shift_center_stack_cpu(image);
	}
	else
	{
		throw std::logic_error("Support for device not implemented");
	}
}
