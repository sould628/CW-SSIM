#pragma once
#include<vector>

namespace SSIM {
	struct Wavelet {
		int width;
		int numPoints;
		std::vector<double> data;
	};

	//a = scalar (width of the wavelet)
	struct Ricker : Wavelet {
		Ricker(const int numPoints, const int a);
	};
}