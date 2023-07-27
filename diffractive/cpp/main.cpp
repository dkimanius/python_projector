#include <torch/script.h>
#include <torch/extension.h>
#include <pybind11/numpy.h>

torch::Tensor trilinear_real_projection_forward(
	torch::Tensor image,
	torch::Tensor rot_matrix);
	
std::vector<torch::Tensor> trilinear_real_projection_backward(
	torch::Tensor image,
	torch::Tensor rot_matrix,
	torch::Tensor dcost_doutput);

torch::Tensor trilinear_complex_projection_forward(
	torch::Tensor image,
	torch::Tensor rot_matrix);
	
std::vector<torch::Tensor> trilinear_complex_projection_backward(
	torch::Tensor image,
	torch::Tensor rot_matrix,
	torch::Tensor dcost_doutput);

void writeComplexVTK(
	torch::Tensor image,
	const std::string fn);
	
void shift_center_3d(
	torch::Tensor image);
	
void shift_center_stack(
	torch::Tensor image);
	

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("real_forward", 
		&trilinear_real_projection_forward, 
		"Trilinear real projector forward",
		py::arg("image"),
		py::arg("rot_matrix"));

	m.def("real_backward", 
		&trilinear_real_projection_backward, 
		"Trilinear real projector backward",
		py::arg("image"),
		py::arg("rot_matrix"),
		py::arg("grad_in"));
	
	m.def("complex_forward", 
		&trilinear_complex_projection_forward, 
		"Trilinear complex projector forward",
		py::arg("image"),
		py::arg("rot_matrix"));

	m.def("complex_backward", 
		&trilinear_complex_projection_backward, 
		"Trilinear complex projector backward",
		py::arg("image"),
		py::arg("rot_matrix"),
		py::arg("grad_in"));
	
	m.def("write_complex_VTK", 
		&writeComplexVTK, 
		"write a complex image in VTK format", 
		py::arg("image"),
		py::arg("filename"));
	
	m.def("shift_center_3d", 
		&shift_center_3d, 
		"shift the center of a 3D Fourier-space image by multiplying with a checkerboard pattern", 
		py::arg("image"));
	
	m.def("shift_center_stack", 
		&shift_center_stack, 
		"shift the centers of a stack of 2D Fourier-space image by multiplying with a checkerboard pattern", 
		py::arg("image"));
}
