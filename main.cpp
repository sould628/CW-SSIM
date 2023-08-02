#include <iostream>
#include <format>
#include <string>
#include <string_view>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>
#include <vector>
#include <algorithm>

#include <armadillo>

//#include "convolution.h"

#include "ssim.h"

#include "wavelet.h"

#include "cwt.h"

using namespace std;

using armVec = arma::vec;

int main() {
	std::cout << "welcome to ssim\n";
	std::cout << std::format("opencv version: {}\n", cv::getVersionString());
	std::cout << std::format("armadillo version: {}\n", arma::arma_version::as_string());

	//armVec v1(1000000, arma::fill::randn);
	//armVec v2(200, arma::fill::randu);

	//armVec v3 = arma::conv(v1, v2, "same");
	//

	//SSIM::Ricker ricker(19, 2.);
	//int width = 30;
	//vector<double> test_array(100);
	//iota(test_array.begin(), test_array.end(), 1);
	//auto conved_array = SSIM::conv1D(test_array, ricker.data);

	//auto cwt_result = SSIM::cwt<SSIM::Ricker>(test_array, width);

	std::string image_path1 = cv::samples::findFile("starry_night.jpg");
	std::string image_path2 = cv::samples::findFile("starry_night.jpg");

	cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_COLOR);
	//cv::imshow("original image", img1);
	if (img1.empty())
	{
		std::cout << "Could not read the image: " << image_path1 << std::endl;
		return 1;
	}
	auto cw_ssim_value_same = SSIM::cw_ssim(img1, img2);
	cout << std::format("cw_ssim of same image: {}\n", cw_ssim_value_same);
	
	//int k = cv::waitKey(0); // Wait for a keystroke in the window
	//if (k == 's')
	//{
	//	imwrite("starry_night.png", img1);
	//}


	return 0;
}