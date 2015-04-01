#ifndef VESSEL_FEATURES_H
#define VESSEL_FEAUTRES_H

#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <string>

using namespace cv;

#define MAT_TYPE CV_64FC1




typedef double e_t; // defines matrix element type

namespace vf
{

	const int width = 640;
	const int height = 480;
	const e_t ksizeHessian = 5;


	struct FrangiFilterOptions {
		int scaleRangeLow, scaleStep, scaleRangeHigh;
		e_t frangiBetaOne, frangiBetaTwo;
		bool verbose;
		bool blackWhite;
		int sigmaNum(){ return (scaleRangeHigh - scaleRangeLow) / scaleStep + 1; }

	};

	static FrangiFilterOptions const defaultFrangiFilterOptions = {
		4, 1, 6,
		0.5, 15.0,
		true,
		true
	};


	// assert((options.scaleRangeHigh - options.scaleRangeLow) % options.scaleStep == 0);


	void Hessian2D(const Mat& In, int Sigma, Mat &Dxx, Mat &Dyy, Mat &Dxy);
	void Eig2Image(const Mat& Dxx, const Mat& Dxy, const Mat& Dyy, Mat& Lambda1, Mat& Lambda2, Mat &Ix, Mat &Iy, Mat &mu1, Mat &mu2);
	void FrangiFilter2D(Mat& In, Mat& Out, Mat& hessianAngle, Mat& hessianDeterminant, FrangiFilterOptions options = defaultFrangiFilterOptions);
	void RidgenessDetection(Mat In, Mat &ridgeness);
	void Ridgeness(Mat& In, int specularThreshold, Mat& ridgenessImage, Mat& hessianDeterminant, Mat& hessianAngle);
	void CalculateRidgeness(Mat& originalImage, Mat& vesselEnhancedImage, Mat& Direction, int sigma, Mat& Out); //S* Review paramters
	void RemoveSpecular(Mat& In, int specularThreshold);

	void Demo(const std::string& testImage);
	void GetGreenChannelAsDouble(const Mat& In, Mat& Out);
	void Imfilter(const Mat& src, Mat& dst, const Mat& ker);
}

#endif // VESSEL_FEAUTRES_H
