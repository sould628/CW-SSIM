#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
namespace SSIM {
	template<typename T>
	inline std::vector<T> conjugate(const std::vector<T>& in);

	template<>
	inline std::vector<double> conjugate(const std::vector<double>& in) {
		std::vector<double> out(in.size());
		std::transform(in.crbegin(), in.crend(), out.begin(), [](const double d) {return d; });
		return out;
	}

	//returns convolution result same size as in1 centered at the fully convoluted output
	template <typename T>
	inline std::vector<T> conv1D(const std::vector<T>& in1, const std::vector<T>& in2);

	template<>
	inline std::vector<double> conv1D<double>(const std::vector<double>& in1, const std::vector<double>& in2) {
		const auto& in2_conjugate = in2;
		const auto full_conv_size = in1.size() + in2_conjugate.size() - 1;
		std::vector<double> full_conv(full_conv_size);
		for (int i = 0; i < full_conv_size; i++) {
			auto in2_start = std::max(0, static_cast<int>(i - in1.size() + 1));
			auto in2_end = std::min(i + 1, static_cast<int>(in2_conjugate.size()));
			auto in1_start = std::min(i, static_cast<int>(in1.size() - 1));
			for (int j = in2_start; j < in2_end; j++) {
				full_conv[i] += in1[in1_start--] * in2_conjugate[j];
			}
		}
		cv::Mat in1_mat(in1);
		cv::Mat in2_mat(in2);
		std::vector<double> out(in1.size());
		const int start_idx_offset = full_conv_size/2 - (in1.size()+1) / 2;
		std::copy(std::next(full_conv.cbegin(), start_idx_offset), std::next(full_conv.cbegin(), start_idx_offset + in1.size()), out.begin());

		return out;
	}

	inline cv::Mat conv1D(const cv::Mat& in1, const cv::Mat& in2) {
		cv::Mat out;
		cv::filter2D(in1, out, -1, in2, cv::Point(-1, -1), 0.0, cv::BORDER_ISOLATED);
		return out;
		//const auto& in2_conjugate = in2;
		//const auto full_conv_size = in1.size() + in2_conjugate.size() - 1;
		//std::vector<double> full_conv(full_conv_size);
		////for (int i = 0; i < full_conv_size; i++) {
		////	auto in2_start = std::max(0, static_cast<int>(i - in1.size() + 1));
		////	auto in2_end = std::min(i + 1, static_cast<int>(in2_conjugate.size()));
		////	auto in1_start = std::min(i, static_cast<int>(in1.size() - 1));
		////	for (int j = in2_start; j < in2_end; j++) {
		////		full_conv[i] += in1[in1_start--] * in2_conjugate[j];
		////	}
		////}
		//cv::Mat in1_mat(in1);
		//cv::Mat in2_mat(in2);
		//std::vector<double> out(in1.size());
		//const int start_idx_offset = full_conv_size / 2 - (in1.size() + 1) / 2;
		//std::copy(std::next(full_conv.cbegin(), start_idx_offset), std::next(full_conv.cbegin(), start_idx_offset + in1.size()), out.begin());

		return out;
	}
}