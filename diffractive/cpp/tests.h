void writeComplexVTK(
	torch::Tensor image,
	const std::string fn);
	
#define FWD_DIFF true

template <typename scalar_t>
complex_t interpolate(
	const torch::TensorAccessor<complex_t, 3> image_in,
	const int size_in,
	scalar_t xp,
	scalar_t yp,
	scalar_t zp)
{
	scalar_t conj = 1;
	
	if (xp < 0)
	{
		xp = -xp;
		yp = -yp;
		zp = -zp;
		
		conj = -1;
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
		
	const complex_t v000 = image_in[z0][y0][x0];
	const complex_t v001 = image_in[z1][y0][x0];
	const complex_t v010 = image_in[z0][y1][x0];
	const complex_t v011 = image_in[z1][y1][x0];
	const complex_t v100 = image_in[z0][y0][x1];
	const complex_t v101 = image_in[z1][y0][x1];
	const complex_t v110 = image_in[z0][y1][x1];
	const complex_t v111 = image_in[z1][y1][x1];

	const complex_t vx00 = LIN_INTERP(fx, v000, v100);
	const complex_t vx10 = LIN_INTERP(fx, v010, v110);
	const complex_t vx01 = LIN_INTERP(fx, v001, v101);
	const complex_t vx11 = LIN_INTERP(fx, v011, v111);
	
	const complex_t vxy0 = LIN_INTERP(fy, vx00, vx10);
	const complex_t vxy1 = LIN_INTERP(fy, vx01, vx11);
	
	const complex_t vxyz = LIN_INTERP(fz, vxy0, vxy1);

	return complex_t(vxyz.real(), conj * vxyz.imag());
}

template <typename scalar_t>
void compute_dout_dp(
	const torch::TensorAccessor<complex_t, 3> image_in,
	const int size_in,
	scalar_t xp,
	scalar_t yp,
	scalar_t zp,
	complex_t* dout_dp)
{
	scalar_t conj = 1;
	
	if (xp < 0)
	{
		xp = -xp;
		yp = -yp;
		zp = -zp;
		
		conj = -1;
	}
	
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
		
		const complex_t voxel = image_in[zk][yk][xk];
		
		dout_dp[0] += (2*ix - 1) * fys[iy] * fzs[iz] * voxel;
		dout_dp[1] += fxs[ix] * (2*iy - 1) * fzs[iz] * voxel;
		dout_dp[2] += fxs[ix] * fys[iy] * (2*iz - 1) * voxel;
	}
	
	// account for the sign change in (xp,yp,zp)
	for (int k = 0; k < 3; k++)
	{
		dout_dp[k] = conj * dout_dp[k];
	}
}

template <typename scalar_t>
void test_00(
	const torch::TensorAccessor<complex_t, 3> image_in,
	const int size_in,
	const scalar_t *rot_matrix,
	const int batch_size,
	torch::TensorAccessor<complex_t, 3> image_out,
	const int size_out)
{
	const int size_out_half = size_out / 2 + 1;	
	const int max_r2 = (size_in/2)*(size_in/2);	
	
	torch::TensorOptions options = torch::TensorOptions().
		dtype(at::ScalarType::ComplexFloat).
		layout(torch::kStrided);
	
	torch::Tensor dV_dX_a = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
		
	torch::Tensor dV_dY_a = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
		
	torch::Tensor dV_dZ_a = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
	
	torch::Tensor dV_dX_n = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
		
	torch::Tensor dV_dY_n = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
		
	torch::Tensor dV_dZ_n = torch::zeros(
		{batch_size, size_out, size_out_half}, options);
		
	torch::TensorAccessor<complex_t, 3> dV_dX_a_acc = dV_dX_a.accessor<complex_t, 3>();
	torch::TensorAccessor<complex_t, 3> dV_dY_a_acc = dV_dY_a.accessor<complex_t, 3>();
	torch::TensorAccessor<complex_t, 3> dV_dZ_a_acc = dV_dZ_a.accessor<complex_t, 3>();
	torch::TensorAccessor<complex_t, 3> dV_dX_n_acc = dV_dX_n.accessor<complex_t, 3>();
	torch::TensorAccessor<complex_t, 3> dV_dY_n_acc = dV_dY_n.accessor<complex_t, 3>();
	torch::TensorAccessor<complex_t, 3> dV_dZ_n_acc = dV_dZ_n.accessor<complex_t, 3>();
		
	const scalar_t eps = 1e-5;
	
	for (int b  = 0; b  < batch_size;    b++)
	for (int yi = 0; yi < size_out;      yi++)
	for (int xi = 0; xi < size_out_half; xi++)
	{		
		const scalar_t x = xi;
		const scalar_t y = yi < size_out/2? yi : yi - size_out;
		
		const int m = 9 * b;			
		scalar_t xp = rot_matrix[m+0] * x + rot_matrix[m+3] * y;
		scalar_t yp = rot_matrix[m+1] * x + rot_matrix[m+4] * y;
		scalar_t zp = rot_matrix[m+2] * x + rot_matrix[m+5] * y;
		
		if (xp*xp + yp*yp + zp*zp <= max_r2)
		{
			#if FWD_DIFF
			
			complex_t V0  = interpolate(image_in, size_in, xp,       yp,       zp);
			complex_t Vxp = interpolate(image_in, size_in, xp + eps, yp,       zp);
			complex_t Vyp = interpolate(image_in, size_in, xp,       yp + eps, zp);
			complex_t Vzp = interpolate(image_in, size_in, xp,       yp,       zp + eps);
			
			dV_dX_n_acc[b][yi][xi] = (Vxp - V0) / eps;
			dV_dY_n_acc[b][yi][xi] = (Vyp - V0) / eps;
			dV_dZ_n_acc[b][yi][xi] = (Vzp - V0) / eps;
			
			#else
			
			complex_t Vxp = interpolate(image_in, size_in, xp + eps, yp,       zp);
			complex_t Vxn = interpolate(image_in, size_in, xp - eps, yp,       zp);
			complex_t Vyp = interpolate(image_in, size_in, xp,       yp + eps, zp);
			complex_t Vyn = interpolate(image_in, size_in, xp,       yp - eps, zp);
			complex_t Vzp = interpolate(image_in, size_in, xp,       yp,       zp + eps);
			complex_t Vzn = interpolate(image_in, size_in, xp,       yp,       zp - eps);
			
			dV_dX_n_acc[b][yi][xi] = (Vxp - Vxn) / (2 * eps);
			dV_dY_n_acc[b][yi][xi] = (Vyp - Vyn) / (2 * eps);
			dV_dZ_n_acc[b][yi][xi] = (Vzp - Vzn) / (2 * eps);
			
			#endif
			
			complex_t dout_dp[] = {0,0,0};
			
			compute_dout_dp(image_in, size_in, xp, yp, zp, dout_dp);
			
			dV_dX_a_acc[b][yi][xi] = dout_dp[0];
			dV_dY_a_acc[b][yi][xi] = dout_dp[1];
			dV_dZ_a_acc[b][yi][xi] = dout_dp[2];
		}
	}
	
	writeComplexVTK(dV_dX_a, "analytic_x.vtk");
	writeComplexVTK(dV_dY_a, "analytic_y.vtk");
	writeComplexVTK(dV_dZ_a, "analytic_z.vtk");
	
	writeComplexVTK(dV_dX_n, "numerical_x.vtk");
	writeComplexVTK(dV_dY_n, "numerical_y.vtk");
	writeComplexVTK(dV_dZ_n, "numerical_z.vtk");
}	
