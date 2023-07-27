#include <vector>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/script.h>
#include <torch/extension.h>

#include "gpu_checks.h"


#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))


template <typename scalar_t>
__global__ void trilinear_real_projection_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image_in,
    const int size_in,
    const scalar_t* __restrict__ rot_matrix,
    const int batch_size,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image_out,
    const int size_out
)
{
	const int max_r2 = (size_in/2)*(size_in/2);
	
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    const int b  = blockIdx.z * blockDim.z + threadIdx.z;
	
	// Cuda threads can only be launched in NxN blocks: 
	// some threads do not correspond to valid pixels.
	if (xi >= size_out || yi >= size_out || b >= batch_size) return;

	// Map image pixels (xi,yi) to 2D Fourier coordinates (x,y):
	const scalar_t x = xi;
	const scalar_t y = yi < size_out/2? yi : yi - size_out;

	// Rotate (x,y) into 3D Fourier coordinates	(xp,yp,zp):
	const int m = 9 * b;
	scalar_t xp = rot_matrix[m+0] * x + rot_matrix[m+3] * y;
	scalar_t yp = rot_matrix[m+1] * x + rot_matrix[m+4] * y;
	scalar_t zp = rot_matrix[m+2] * x + rot_matrix[m+5] * y;

	// Interpolate between the 8 closest voxels:
	if (xp*xp + yp*yp + zp*zp <= max_r2)
	{
		// Friedel symmetry:
		// We only store the non-negative half of the image.
		// If xp is negative, consider the conjugate of the 
		// value at the opposing position. Since this is the 
		// real-valued implementation, we omit the conjugation.
		
		if (xp < 0)
		{
			xp = -xp;
			yp = -yp;
			zp = -zp;
		}
	
		int x0 = floor(xp);
		int y0 = floor(yp);
		int z0 = floor(zp);
		
		const scalar_t fx = xp - x0;
		const scalar_t fy = yp - y0;
		const scalar_t fz = zp - z0;
		
		if (y0 < 0) y0 += size_in;
		if (z0 < 0) z0 += size_in;
		
		const int x1 = std::min(x0 + 1, size_in - 1);
		const int y1 = (y0 + 1) % size_in;
		const int z1 = (z0 + 1) % size_in;
			
		const scalar_t v000 = image_in[z0][y0][x0];
		const scalar_t v001 = image_in[z1][y0][x0];
		const scalar_t v010 = image_in[z0][y1][x0];
		const scalar_t v011 = image_in[z1][y1][x0];
		const scalar_t v100 = image_in[z0][y0][x1];
		const scalar_t v101 = image_in[z1][y0][x1];
		const scalar_t v110 = image_in[z0][y1][x1];
		const scalar_t v111 = image_in[z1][y1][x1];

		const scalar_t vx00 = LIN_INTERP(fx, v000, v100);
		const scalar_t vx10 = LIN_INTERP(fx, v010, v110);
		const scalar_t vx01 = LIN_INTERP(fx, v001, v101);
		const scalar_t vx11 = LIN_INTERP(fx, v011, v111);
		
		const scalar_t vxy0 = LIN_INTERP(fy, vx00, vx10);
		const scalar_t vxy1 = LIN_INTERP(fy, vx01, vx11);
		
		image_out[b][yi][xi] = LIN_INTERP(fz, vxy0, vxy1);
	}
}

template <typename scalar_t>
__global__ void trilinear_real_projection_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> image_in,
    const int size_in,
    const scalar_t* __restrict__ rot_matrix,
    const int batch_size,
    const int size_out,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dcost_doutput,
	      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dcost_dimage_in,
    scalar_t* __restrict__ dcost_drot_matrix )
{
    
	const int max_r2 = (size_in/2)*(size_in/2);
	
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    const int b  = blockIdx.z * blockDim.z + threadIdx.z;
	
	// Cuda threads can only be launched in NxN blocks: 
	// some threads do not correspond to valid pixels.
	if (xi >= size_out || yi >= size_out || b >= batch_size) return;

	// Map image pixels (xi,yi) to 2D Fourier coordinates (x,y):
	const scalar_t x = xi;
	const scalar_t y = yi < size_out/2? yi : yi - size_out;
		
	const int m = 9 * b;
	
	scalar_t xp = rot_matrix[m+0] * x + rot_matrix[m+3] * y;
	scalar_t yp = rot_matrix[m+1] * x + rot_matrix[m+4] * y;
	scalar_t zp = rot_matrix[m+2] * x + rot_matrix[m+5] * y;

	if (xp*xp + yp*yp + zp*zp <= max_r2)
	{
		scalar_t conj = 1;
		
		if (xp < 0)
		{
			xp = -xp;
			yp = -yp;
			zp = -zp;
			
			conj = -1;
		}
		
		// 'C' is the current cost, 'out' is the output value for that pixel
		scalar_t dC_dout = dcost_doutput[b][yi][xi];
		
		// dout_dp means [d(out)/d(xp), d(out)/d(yp), d(out)/d(zp)]
		scalar_t dout_dp[] = {0,0,0};
		
		int x0 = floor(xp);
		int y0 = floor(yp);
		int z0 = floor(zp);
		
		const scalar_t fx = xp - x0;
		const scalar_t fy = yp - y0;
		const scalar_t fz = zp - z0;
			
		if (y0 < 0) y0 += size_in;
		if (z0 < 0) z0 += size_in;

		scalar_t fxs[] = {(scalar_t) 1.0 - fx, fx};
		scalar_t fys[] = {(scalar_t) 1.0 - fy, fy};
		scalar_t fzs[] = {(scalar_t) 1.0 - fz, fz};
			
		for (int k = 0; k < 8; k++)
		{
			//   k =   0    1    2    3    4    5    6    7 
			// xyz = 000, 100, 010, 110, 001, 101, 011, 111
			
			const int ix =  k%2;
			const int iy = (k/2)%2;
			const int iz =  k/4;
			
			const int xk = std::min(x0 + ix, size_in - 1);
			const int yk = (y0 + iy) % size_in;
			const int zk = (z0 + iz) % size_in;
			
			const scalar_t voxel = image_in[zk][yk][xk];
			const scalar_t w_k = fxs[ix] * fys[iy] * fzs[iz];
			
			// out = sum_k( w_k * voxel_k )  =>  d(out)/d(voxel_k) = w_k
			
			atomicAdd(&dcost_dimage_in[zk][yk][xk], dC_dout * w_k);
			
			dout_dp[0] += (2*ix - 1) * fys[iy] * fzs[iz] * voxel;
			dout_dp[1] += fxs[ix] * (2*iy - 1) * fzs[iz] * voxel;
			dout_dp[2] += fxs[ix] * fys[iy] * (2*iz - 1) * voxel;
		}
			
		// account for the sign change in (xp,yp,zp)
		for (int k = 0; k < 3; k++)
		{
			dout_dp[k] = conj * dout_dp[k];
		}	
			
		for (int k = 0; k < 3; k++)
		{
			const scalar_t dC_dpk = dC_dout * dout_dp[k];
		
			// p = Ax  =>  dp[j] / dA[ji] = x[i]
			
			atomicAdd(&dcost_drot_matrix[m + k    ], dC_dpk * x);
			atomicAdd(&dcost_drot_matrix[m + k + 3], dC_dpk * y);
		}
    }
}

void trilinear_real_projection_forward_cuda(
    const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
	torch::Tensor image_out
)
{
	CHECK_INPUT(image_in)
	CHECK_INPUT(rot_matrix)
	CHECK_INPUT(image_out)
	
	const int batch_size = rot_matrix.size(0);
	const int size_in    = image_in.size(0);
	const int size_out   = size_in;
	
    const int threads = 8;
	
	const int num_blocks_xy = (int)ceil(size_out / (double)threads);
	const int num_blocks_b  = (int)ceil(batch_size / (double)threads);
	
	dim3 dimBlock(threads, threads, threads);
    dim3 dimGrid(num_blocks_xy, num_blocks_xy, num_blocks_b);

    AT_DISPATCH_FLOATING_TYPES(
        rot_matrix.scalar_type(),
        "trilinear_real_projection_forward_cuda_kernel",
        ([&] {
            trilinear_real_projection_forward_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
					image_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
					size_in,
					rot_matrix.data_ptr<scalar_t>(),
					rot_matrix.size(0),
					image_out.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
					size_out
            );
        })
    );
}

void trilinear_real_projection_backward_cuda(
    const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
    const torch::Tensor dcost_doutput,
          torch::Tensor dcost_dimage_in,
          torch::Tensor dcost_drot_matrix
)
{
    CHECK_INPUT(image_in)
	CHECK_INPUT(rot_matrix)
    CHECK_INPUT(dcost_doutput)
    CHECK_INPUT(dcost_dimage_in)
    CHECK_INPUT(dcost_drot_matrix)

	const int batch_size = rot_matrix.size(0);
	const int size_in    = image_in.size(0);
	const int size_out   = size_in;
	
    const int threads = 8;
	
	const int num_blocks_xy = (int)ceil(size_out / (double)threads);
	const int num_blocks_b  = (int)ceil(batch_size / (double)threads);
	
	dim3 dimBlock(threads, threads, threads);
    dim3 dimGrid(num_blocks_xy, num_blocks_xy, num_blocks_b);

    AT_DISPATCH_FLOATING_TYPES(
        rot_matrix.scalar_type(),
        "trilinear_real_projection_backward_cuda_kernel",
        ([&] {
            trilinear_real_projection_backward_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
				image_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
				size_in,
				rot_matrix.data_ptr<scalar_t>(),
				rot_matrix.size(0),
				size_out,
                dcost_doutput.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                dcost_dimage_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                dcost_drot_matrix.data_ptr<scalar_t>());
        })
    );
}
