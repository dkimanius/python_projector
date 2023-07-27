#include <torch/script.h>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <pybind11/numpy.h>


void writeComplexVTK(
	torch::Tensor image,
	const std::string fn)
{
	const int d = image.size(0);
	const int h = image.size(1);
	const int w = image.size(2);
	
	std::ofstream os(fn.c_str(), std::ios::binary);

    std::string sizetype = "float";
	const size_t size = w * h * d;

    os << "# vtk DataFile Version 2.0\n";
    os << "Volume example\n";

    os << "ASCII\n";

    os << "DATASET STRUCTURED_POINTS\n";
    os << "DIMENSIONS " << w << " " << h << " " << d << "\n";
    os << "SPACING " << 1 << " " << 1 << " " << 1 << "\n";
    os << "ORIGIN " << 0 << " " << 0 << " " << 0 << "\n";
    os << "POINT_DATA " << size << "\n";
    os << "SCALARS volume_scalars " << sizetype << " 2\n";
    os << "LOOKUP_TABLE default\n";
	
	const torch::TensorAccessor<c10::complex<float>, 3> acc = image.accessor<c10::complex<float>, 3>();
	
	for (int z = 0; z < d; z++)
	for (int y = 0; y < h; y++)
	for (int x = 0; x < w; x++)
	{
		os << acc[z][y][x].real() << "\n";
		os << acc[z][y][x].imag() << "\n";
	}
	
	os.flush();
}
