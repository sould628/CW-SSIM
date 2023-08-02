#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <armadillo>
//#include "convolution.h"
#include <opencv2/imgproc.hpp>
#include "wavelet.h"

namespace SSIM {
	template<typename Wavelet_Type>
	std::vector<std::vector<double>> cwt(std::vector<double>& data, const double width) {
		using namespace std;

		vector<int> widths(width);
		iota(widths.begin(), widths.end(), 1);
		vector<vector<double>> output;
		output.reserve(width);

		for (int i = 0; i < widths.size(); i++) {
			const int width = widths[i];
			const int N = std::min(10 * width, static_cast<int>(data.size()));
			Wavelet_Type wavelet(N, width);
			vector<double> ith_out;
			arma::vec sig(data);
			arma::vec wav(wavelet.data);
			arma::vec convoluted = arma::conv(sig, wav);
			std::vector<double> full_conv = arma::conv_to<std::vector<double>>::from(convoluted);
			const int full_conv_size = full_conv.size();
			const int start_idx_offset = full_conv_size / 2 - (data.size() + 1) / 2;
			std::vector<double> same_conv_size(data.size());
			std::copy(std::next(full_conv.cbegin(), start_idx_offset), std::next(full_conv.cbegin(), start_idx_offset + data.size()), same_conv_size.begin());

			//output.push_back(arma::conv_to<std::vector<double>>::from(convoluted));
			output.push_back(same_conv_size);
		}
		return output;
	}
	template<typename Wavelet_Type>
	cv::Mat cwt(const cv::Mat& data, const double width) {
		using namespace std;

		vector<int> widths(width);
		iota(widths.begin(), widths.end(), 1);
		vector<vector<double>> output(widths.size());

		cv::Mat out(cv::Size(data.size().area(), width), data.type());

		for (int i = 0; i < widths.size(); i++) {
			const int width = widths[i];
			const int N = std::min(10 * width, static_cast<int>(data.size().area()));
			Wavelet_Type wavelet(N, width);
			//cout << data << endl;
			cv::Mat convResult;
			cv::filter2D(data, convResult, -1, cv::Mat(wavelet.data), cv::Point(-1, -1), 0.0, cv::BORDER_ISOLATED);
			cout << convResult.colRange(0, 10);
			//cout << convResult << endl;
			//auto convResult = conv1D(data, cv::Mat(wavelet.data));
			out.row(i) = convResult.clone();
			//if (out.empty())
			//	out = convResult;
			//else
			//	out.push_back(convResult);
			//cout << output_test <<endl;
			//output[i] = conv1D(data, wavelet.data);
		}
		return out;
	}

}