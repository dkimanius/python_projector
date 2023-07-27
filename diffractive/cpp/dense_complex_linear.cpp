#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

void trilinear_complex_projection_forward_cpu(
	const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
	torch::Tensor image_out);
	
void trilinear_complex_projection_forward_cuda(
    const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
	torch::Tensor image_out);
	
void trilinear_complex_projection_backward_cpu(
    const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
    const torch::Tensor dcost_doutput,
          torch::Tensor dcost_dimage_in,
          torch::Tensor dcost_drot_matrix);
	
void trilinear_complex_projection_backward_cuda(
    const torch::Tensor image_in,
	const torch::Tensor rot_matrix,
    const torch::Tensor dcost_doutput,
          torch::Tensor dcost_dimage_in,
          torch::Tensor dcost_drot_matrix);
		  
void writeComplexVTK(
	torch::Tensor image,
	const std::string fn);
	
void shift_center_3d(
	torch::Tensor image);
	
void shift_center_stack(
	torch::Tensor image);		


torch::Tensor trilinear_complex_projection_forward(
	torch::Tensor image,
	torch::Tensor rot_matrix)
{
	const int s = image.size(0);
	const int sh = image.size(2);
	const int b = rot_matrix.size(0);
	
	if (rot_matrix.size(1) != 3 || rot_matrix.size(2) != 3)
	{
		throw std::logic_error("Rotation matrices have to be 3x3");
	}
	
	auto output = torch::zeros(
		{b, s, sh},
		torch::TensorOptions()
			.dtype(image.dtype())
			.device(image.device())
			.layout(torch::kStrided)
			.requires_grad(true)
	);
	
	if (image.device().type() == torch::kCPU)
	{
		trilinear_complex_projection_forward_cpu(image, rot_matrix, output);
	}
	else
	{
		trilinear_complex_projection_forward_cuda(image, rot_matrix, output);
	}

	return output;
}


std::vector<torch::Tensor> trilinear_complex_projection_backward(
	torch::Tensor image,
	torch::Tensor rot_matrix,
	torch::Tensor dcost_doutput)
{
	const int s = image.size(0);
	const int sh = image.size(2);
	const int b = rot_matrix.size(0);

	auto grad_image = torch::zeros(
		{s, s, sh},
		torch::TensorOptions()
			.dtype(image.dtype())
			.device(image.device())
			.layout(torch::kStrided)
			.requires_grad(false)
	);

	auto grad_rot_matrix = torch::zeros(
		{b, 3, 3},
		torch::TensorOptions()
			.dtype(rot_matrix.dtype())
			.device(rot_matrix.device())
			.layout(torch::kStrided)
			.requires_grad(false)
	);

	if (image.device().type() == torch::kCPU)
	{
		trilinear_complex_projection_backward_cpu(
			image,
			rot_matrix,
			dcost_doutput,
			grad_image,
			grad_rot_matrix
		);
	}
	else
	{
		trilinear_complex_projection_backward_cuda(
			image,
			rot_matrix,
			dcost_doutput,
			grad_image,
			grad_rot_matrix
		);
	}
	
	return {grad_image, grad_rot_matrix};
}
