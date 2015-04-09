#ifndef VESSEL_FEATURES_H
#define VESSEL_FEAUTRES_H

#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <string>

using namespace cv;

#define MAT_TYPE_ID CV_64FC1

typedef double e_t; // defines matrix element type

namespace vf
{
	struct FrangiFilterOptions {
		int scaleRangeLow;
		int scaleStep;
		int scaleRangeHigh;
		e_t frangiBetaOne;
		e_t frangiBetaTwo;
		bool verbose;
		bool blackWhite;
		int sigmaNum(){ return (scaleRangeHigh - scaleRangeLow) / scaleStep + 1; }
	};
	/* Matlab Function Call Hierarchy
	Demo
		detector_RBCT
			ridgeness
				FrangiFilter x
					hessian x
					eig2image x
				calc_ridgeness
			RBCT
		detector_RBSD
	
	
	
	*/


	const FrangiFilterOptions FRANGI_DEFAULT {4, 1, 6, 0.5, 15.0, true, true};

	void Hessian2D(const Mat& In, int Sigma, Mat &Dxx, Mat &Dyy, Mat &Dxy);
	void Eig2Image(const Mat& Dxx, const Mat& Dxy, const Mat& Dyy, Mat& Lambda1, Mat& Lambda2, Mat &Ix, Mat &Iy, Mat &mu1, Mat &mu2);
	void FrangiFilter2D(Mat& In, Mat& Out, Mat& hessianAngle, Mat& hessianDeterminant, FrangiFilterOptions options = FRANGI_DEFAULT);
	void RidgenessDetection(Mat In, Mat &ridgeness, const e_t kSizeHessian=5.0);
	void Ridgeness(Mat& In, int specularThreshold, Mat& ridgenessImage, Mat& hessianDeterminant, Mat& hessianAngle);
	void CalculateRidgeness(Mat& originalImage, Mat& enhancedIn, Mat& Direction, int sigma, Mat& Out); //S* Review paramters
	void RemoveSpecular(Mat& In, int specularThreshold);

	void Demo(const std::string& testImage);
	void GetGreenChannelAsDouble(const Mat& In, Mat& Out);
	void Imfilter(const Mat& src, Mat& dst, const Mat& ker);
}

#endif // VESSEL_FEAUTRES_H
