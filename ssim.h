#pragma once

#include <opencv2/core.hpp>

namespace SSIM {
	struct SSIM_PARAMETERS {
		double L{ 255. };
		double k1{ 0.01 };
		double k2{ 0.03 };
		double k{ 0.01 };
	} static const default_ssim_parameters;

	double cw_ssim(const cv::Mat& img1, const cv::Mat& img2, const SSIM_PARAMETERS& ssim_params = default_ssim_parameters, const int kernel_width = 30);
}