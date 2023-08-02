#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <numeric> //iota
#include <algorithm> //for_each
#include "wavelet.h"


namespace SSIM {
	using namespace std;
	//a.k.a Mexican hat wavelet
	Ricker::Ricker(const int _numPoints, const int _a) :Wavelet{ _a, _numPoints, vector<double>(numPoints) }
	{
		auto& data = Wavelet::data;
		const double numPoints = static_cast<double>(_numPoints);
		const double a = static_cast<double>(_a);
		const double A = 2. / (sqrt(3. * a) * pow(M_PI, 0.25));
		const double wsq = a * a;

		vector<double> vec(numPoints);
		iota(vec.begin(), vec.end(), 0);
		for_each(vec.begin(), vec.end(), [&numPoints](double& d) {d = d - (numPoints - 1.0) / 2.; });

		vector<double> xsq(numPoints);
		transform(vec.cbegin(), vec.cend(), vec.cbegin(), xsq.begin(), multiplies<>{});

		vector<double> mod(numPoints);
		transform(xsq.cbegin(), xsq.cend(), mod.begin(), [&wsq](const double xsq) {return 1. - xsq / wsq; });

		vector<double> gauss(numPoints);
		transform(xsq.cbegin(), xsq.cend(), gauss.begin(), [&wsq](const double xsq) {return exp(-xsq / (2. * wsq)); });

		vector<double> total(numPoints);
		transform(mod.cbegin(), mod.cend(), gauss.cbegin(), total.begin(), [&A](const double mod, const double gauss) {return A * mod * gauss; });

		copy(total.cbegin(), total.cend(), data.begin());
	}
}