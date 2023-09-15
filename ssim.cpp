#define ARMA_PRINT_EXCEPTIONS

#include "ssim.h"
#include "cwt.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <armadillo>

#include <format>

#ifdef NDEBUG
#define BRIEF_PRINT(x)
#else
#define BRIEF_PRINT(x) x.brief_print(#x ":")
#endif

namespace SSIM {
    cv::Mat gaussian_convolution_2d(const cv::Mat& img1, const int gaussian_kernel_width = 11, const double gaussian_kernel_sigma = 1.5) {
        cv::Mat result;
        cv::GaussianBlur(img1, result, cv::Size(gaussian_kernel_width, gaussian_kernel_width), gaussian_kernel_sigma);
        return result;
    }

    double cw_ssim(const cv::Mat& img1, const cv::Mat& img2, const SSIM_PARAMETERS& ssim_params, const int width)
    {
        using namespace std;

        double c1 = ssim_params.k1 * ssim_params.k1 * ssim_params.L * ssim_params.L;
        double c2 = ssim_params.k2 * ssim_params.k2 * ssim_params.L * ssim_params.L;
        double k = ssim_params.k;

        cv::Mat img1_gray, img2_gray;

        cv::cvtColor(img1, img1_gray, cv::COLOR_RGB2GRAY);
        cv::cvtColor(img2, img2_gray, cv::COLOR_RGB2GRAY);

        auto img1_flat = img1_gray.reshape(1, 1);
        auto img2_flat = img2_gray.reshape(1, 1);

        vector<double> img1_vec, img2_vec;

        img1_flat.row(0).copyTo(img1_vec);
        img2_flat.row(0).copyTo(img2_vec);

        auto cwt1 = cwt<Ricker>(img1_vec, width);
        auto cwt2 = cwt<Ricker>(img2_vec, width);
        
        assert(cwt1.size() == cwt2.size());

        //auto cwtMatr1 = cv::Mat(cwt1.size(), cwt1[0].size(), cv::DataType<double>::type);
        //auto cwtMatr2 = cv::Mat(cwt2.size(), cwt2[0].size(), cv::DataType<double>::type);

        //for (int i = 0; i < cwt1.size(); i++) {
        //    cwtMatr1.row(i) = cv::Mat(cwt1[i]);
        //    cwtMatr2.row(i) = cv::Mat(cwt2[i]);
        //}

        //auto cwtMatr1 = cwt<Ricker>(img1_flat, width);
       // cout << cwtMatr1 << endl;
        //auto cwtMatr2 = cwt<Ricker>(img2_flat, width);

        //cout << cwtMatr1.colRange(0, 10)<<endl;
        //cout << cwtMatr2.colRange(0, 10) << endl;



        //cv::Mat c1c2;
        //cv::Mat absCwtMatr1 = cv::abs(cwtMatr1);
        //cv::Mat absCwtMatr2 = cv::abs(cwtMatr2);
        //cv::multiply(absCwtMatr1, absCwtMatr2,c1c2);

        //cout << c1c2.colRange(0, 10) << endl;



        //arma::mat cwtMat1(&cwt1[0][0], cwt1.size(), cwt1[0].size());
        //arma::mat cwtMat2(&cwt2[0][0], cwt2.size(), cwt2[0].size());

        arma::mat cwtMat1(cwt1.size(), cwt1[0].size());
        arma::mat cwtMat2(cwt2.size(), cwt2[0].size());


        for (int i = 0; i < cwt1.size(); i++) {
            cwtMat1.row(i) = arma::conv_to<arma::Row<double>>::from(cwt1[i]);
            cwtMat2.row(i) = arma::conv_to<arma::Row<double>>::from(cwt2[i]);
        }

        arma::mat c1c2 = (arma::mat)abs(cwtMat1) % (arma::mat)abs(cwtMat2);

        BRIEF_PRINT(c1c2);

        /*cv::Mat c1_2, c2_2;*/
        //cv::pow(cwtMatr1, 2, c1_2);
        //cv::pow(cwtMatr2, 2, c2_2);

        arma::mat c1_2 = cwtMat1 % cwtMat1;
        arma::mat c2_2 = cwtMat2 % cwtMat2;


        BRIEF_PRINT(c1_2);
        BRIEF_PRINT(c2_2);

        //cv::Mat num_ssim_1;
        //cv::reduce(c1c2, num_ssim_1, 0, cv::REDUCE_SUM, CV_32SC1);
        //c1c2 = c1c2 * 2. + k;

        arma::mat num_ssim_1 = 2. * arma::sum(c1c2, 0) + k;

        BRIEF_PRINT(num_ssim_1);

        //cv::Mat den_ssim_1, den_ssim_1_1, den_ssim_1_2;

        //cv::reduce(c1_2, den_ssim_1_1, 0, cv::REDUCE_SUM, CV_32SC1);
        //cv::reduce(c2_2, den_ssim_1_2, 0, cv::REDUCE_SUM, CV_32SC1);

        //den_ssim_1 = den_ssim_1_1 + den_ssim_1_2 + k;

        arma::mat den_ssim_1 = arma::sum(c1_2, 0) + arma::sum(c2_2, 0) + k;

        BRIEF_PRINT(den_ssim_1);

        //cv::Mat c1c2_conj;
        //cv::multiply(cwtMatr1, cwtMatr2, c1c2_conj);
        
        arma::mat c1c2_conj = cwtMat1 % arma::conj(cwtMat2);

        BRIEF_PRINT(c1c2_conj);

        //cv::Mat num_ssim_2;
        //cv::reduce(c1c2_conj, num_ssim_2, 0, cv::REDUCE_SUM, CV_32SC1);
        //num_ssim_2 = 2. * num_ssim_2 + k;

        arma::mat num_ssim_2 = 2. * arma::abs(arma::sum(c1c2_conj, 0)) + k;

        BRIEF_PRINT(num_ssim_2);
 
        //cv::Mat den_ssim_2_1 = cv::abs(c1c2_conj);
        //cv::Mat den_ssim_2;
        //cv::reduce(den_ssim_2_1, den_ssim_2, 0, cv::REDUCE_SUM, CV_32SC1);
        //den_ssim_2 = den_ssim_2 + k;

        arma::mat den_ssim_2 = 2. * arma::sum(arma::abs(c1c2_conj), 0) + k;

        BRIEF_PRINT(den_ssim_2);

        //cv::Mat ssim_map_1, ssim_map_2, ssim_map;
        //cv::divide(num_ssim_1, den_ssim_1, ssim_map_1);
        //cv::divide(num_ssim_2, den_ssim_2, ssim_map_2);
        //cv::multiply(ssim_map_1, ssim_map_2, ssim_map);

        arma::mat ssim_map = (num_ssim_1 / den_ssim_1) % (num_ssim_2 / den_ssim_2);

        BRIEF_PRINT(ssim_map);

        //cout << ssim_map << endl;

        double index = arma::mean(arma::mean(ssim_map));

        return index;
    }
}