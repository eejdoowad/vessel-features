#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv/cv.h>
#include <iostream>
#include <cmath>
#include"vessel_features.hpp"

#define VESSEL_FEATURES_PI           3.14159265358979323846

using namespace cv;

namespace vf
{

	// Hessian2D, Complete
	// Input: In, Sigma
	// Output: Dxx, Dyy, Dxy (note that Dyx == Dxy)
	void Hessian2D(const Mat& In, int Sigma, Mat &Dxx, Mat &Dyy, Mat &Dxy)
	{
		const int kern_width = 6 * Sigma + 1;
		// and height, the kernal extends a multiple of 3 out from the center in each direction

		Mat Y = Mat(kern_width, kern_width, MAT_TYPE_ID);

		for (int i = 0, val = -3 * Sigma; i < kern_width; i++, val++)
		{
			e_t * const pY = Y.ptr<e_t>(i);
			for (int j = 0; j < kern_width; j++)
			{
				pY[j] = static_cast<e_t>(val);
			}
		}
		Mat X = Y.t();

		// Create Filters
		Mat exp_term;
		cv::exp(-(X.mul(X) + Y.mul(Y)) / (2 * Sigma * Sigma), exp_term);
		Mat DGaussxx = (1 / (2 * VESSEL_FEATURES_PI * pow(Sigma, 4))) * ((X.mul(X)) / (Sigma * Sigma) - 1).mul(exp_term);
		Mat DGaussxy = (1 / (2 * VESSEL_FEATURES_PI * pow(Sigma, 6))) *  (X.mul(Y)).mul(exp_term);
		Mat DGaussyy = DGaussxx.t();

		// Apply convolution filters
		Imfilter(In, Dxx, DGaussxx);
		Imfilter(In, Dxy, DGaussxy);
		Imfilter(In, Dyy, DGaussyy);
	}

	// Eig2Image, Complete
	// Input: Dxx, Dxy, Dyy
	// Output: Lambda1, Lambda2, Ix, Iy, mu1, mu2
	void Eig2Image(const Mat& Dxx, const Mat& Dxy, const Mat& Dyy, Mat& Lambda1, Mat& Lambda2, Mat& Ix, Mat& Iy, Mat& mu1, Mat& mu2)
	{
		const int rows = Dxx.rows, cols = Dxx.cols;
		for (int i = 0; i<rows; i++)
		{
			e_t const * const pDxx = Dxx.ptr<e_t>(i);
			e_t const * const pDyy = Dyy.ptr<e_t>(i);
			e_t const * const pDxy = Dxy.ptr<e_t>(i);
			e_t * const pLambda1 = Lambda1.ptr<e_t>(i);
			e_t * const pLambda2 = Lambda2.ptr<e_t>(i);
			e_t * const pmu1 = mu1.ptr<e_t>(i);
			e_t * const pmu2 = mu2.ptr<e_t>(i);
			e_t * const pIx = Ix.ptr<e_t>(i);
			e_t * const pIy = Iy.ptr<e_t>(i);

			for (int j = 0; j<cols; j++)
			{
				// Compute eigenvectors
				const e_t dxx = pDxx[j], dyy = pDyy[j], dxy = pDxy[j];
				const e_t tmp = sqrt((dxx - dyy)*(dxx - dyy) + 4 * dxy*dxy);
				e_t v1x = 2 * dxy;
				e_t v1y = dyy - dxx + tmp;

				// normalize eigenvectors
				const e_t mag = sqrt(v1x*v1x + v1y*v1y);
				v1x = v1x / mag;
				v1y = v1y / mag;

				// eigenvectors are orthogonal S: ???
				pIx[j] = -v1y;
				pIy[j] = v1x;

				// compute eigenvalues
				const e_t mu1 = pmu1[j] = (dxx + dyy - tmp) / 2;
				const e_t mu2 = pmu2[j] = (dxx + dyy + tmp) / 2;

				// Sort eigen values by abs
				bool mu1_gt_mu2 = abs(mu1) > abs(mu2);
				pLambda1[j] = (mu1_gt_mu2) ? mu2 : mu1;
				pLambda2[j] = (mu1_gt_mu2) ? mu1 : mu2;
			}
		}
	}

	// FrangiFilter2d, Complete
	// Enhances vessels on image
	// Input: In
	// Output: Out, hessianAngle, hessianDeterminant
	// Note: User must preallocate memory for all matrices, input and output, before calling function
	// Note: Scale output from original Matlab unused, so is  not included
	void FrangiFilter2D(Mat& In, Mat& Out, Mat& hessianAngle, Mat& hessianDeterminant, FrangiFilterOptions options)
	{
		// In: input image, example code specified values in range  0 : 255, we'll see \**
		// Out: output image with enhanced vessels
		// Scale: output matrix containing scales on which maximum intensity of every pixel is found (? whatever that means)
		// hessianAngle: output matrix containing directions (angles) of pixels
		// hessianDeterminant: determinant of Hessian: lambda1 * lambda 2
		const int height = In.rows;
		const int width = In.cols;

		e_t beta = 2 * options.frangiBetaOne * options.frangiBetaOne;
		e_t c = 2 * options.frangiBetaTwo * options.frangiBetaTwo;

		// loop through sigma values /* must rework looping conditions
		for (int sigma = options.scaleRangeLow, k = 0; sigma <= options.scaleRangeHigh; sigma += options.scaleStep, k++)
		{
			if (options.verbose)
			{
				std::cout << "Current Frangi Filter Sigma: " << sigma << std::endl;
			}

			Mat Dxx(height, width, MAT_TYPE_ID);
			Mat Dxy(height, width, MAT_TYPE_ID);
			Mat Dyy(height, width, MAT_TYPE_ID);

			Hessian2D(In, sigma, Dxx, Dyy, Dxy);

			Dxx = (sigma * sigma) * Dxx;
			Dxy = (sigma * sigma) * Dxy;
			Dyy = (sigma * sigma) * Dyy;

			Mat Lambda1(height, width, MAT_TYPE_ID);
			Mat Lambda2(height, width, MAT_TYPE_ID);
			Mat Ix(height, width, MAT_TYPE_ID);
			Mat Iy(height, width, MAT_TYPE_ID);
			Mat Mu1(height, width, MAT_TYPE_ID);
			Mat Mu2(height, width, MAT_TYPE_ID);

			Eig2Image(Dxx, Dxy, Dyy, Lambda1, Lambda2, Ix, Iy, Mu1, Mu2);

			const int rows = In.rows, cols = In.cols;
			for (int i = 0; i < rows; i++)
			{
				e_t const * const pDxx = Dxx.ptr<e_t>(i);
				e_t const * const pDyy = Dyy.ptr<e_t>(i);
				e_t const * const pDxy = Dxy.ptr<e_t>(i);
				e_t * const pLambda1 = Lambda1.ptr<e_t>(i);
				e_t * const pLambda2 = Lambda2.ptr<e_t>(i);
				e_t * const pIx = Ix.ptr<e_t>(i);
				e_t * const pIy = Iy.ptr<e_t>(i);
				e_t * const pMu1 = Mu1.ptr<e_t>(i);
				e_t * const pMu2 = Mu2.ptr<e_t>(i);

				e_t * const pOut = Out.ptr<e_t>(i);
				e_t * const pDir = hessianAngle.ptr<e_t>(i);
				e_t * const pDoH = hessianDeterminant.ptr<e_t>(i);

				for (int j = 0; j < cols; j++)
				{
					e_t curMu1 = pMu1[j];
					e_t curMu2 = pMu2[j];

					if (options.blackWhite)
					{
						if (curMu1 < 0) curMu1 = 0;
						if (curMu2 < 0) curMu2 = 0;
					}
					else
					{
						if (curMu1 > 0) curMu1 = 0;
						if (curMu2 > 0) curMu2 = 0;
					}

					// Compute output image by determining max output pixel value
					// and angle associated with max pixel value
					e_t S2_temp = (curMu1 * curMu1) + (curMu2 * curMu2);
					e_t curOut = (e_t)1.0 - exp(-S2_temp / c);
					e_t curDir = atan2(pIy[j], pIx[j]);
					if (k == 0) // Out from first sigma is default value
					{
						pOut[j] = curOut;
						pDir[j] = curDir;
					}
					else if (pOut[j] < curOut) // otherwise change Out to max value
					{
						pOut[j] = curOut;
						pDir[j] = curDir;
					}

					//set hessianDeterminant to maximum value across all sigmas
					e_t curDoH = curMu1 * curMu2;
					if (k == 0) // hessianDeterminant from first sigma is default value
					{
						pDoH[j] = curDoH;
					}
					else if (pDoH[j] < curDoH) // otherwise change hessianDeterminant to max value
					{
						pDoH[j] = curDoH;
					}
				}
			}
		}
	}


	// Not too sure about this function. Will have to follow up.
	void RidgenessDetection(Mat In, Mat &ridgeness, e_t const kSizeHessian)
	{
		int frangiScaleMin = 3, frangiScaleMax = 5, frangiScaleRatio = 1;
		e_t frangiBetaOne = 0.5, frangiBetaTwo = 15;
		e_t beta = 2 * frangiBetaOne*frangiBetaOne;
		e_t c = 2 * frangiBetaTwo*frangiBetaTwo;
		int cols = In.cols, rows = In.rows;
		Mat Ixx(rows, cols, CV_64F), Iyy(rows, cols, CV_64F), Ixy(rows, cols, CV_64F);//using e_t precision
		Mat mu1(rows, cols, CV_64F), mu2(rows, cols, CV_64F), Vx(rows, cols, CV_64F), Vy(rows, cols, CV_64F);
		// S: temp addition for testing
		Mat Lambda1(rows, cols, CV_64F), Lambda2(rows, cols, CV_64F);
		//

		for (int s = frangiScaleMin; frangiScaleMin + s*frangiScaleRatio <= frangiScaleMax; s++) //* S:Changed commas to semicolons
		{
			Hessian2D(In, s, Ixx, Iyy, Ixy);
			//correct for scale
			Ixx = (kSizeHessian * kSizeHessian) * Ixx;
			Ixy = (kSizeHessian * kSizeHessian) * Ixy;
			Iyy = (kSizeHessian * kSizeHessian) * Iyy;

			// S: Reorder parameters later
			Eig2Image(Ixx, Iyy, Ixy, Lambda1, Lambda2, mu1, mu2, Vx, Vy);
			ridgeness = Ixx;
		}

	}

	// Inputs: In, specularThreshold
	// Outputs: ridgenessImage, hessianDeterminant, hessianAngle
	void Ridgeness(Mat& In, int specularThreshold, Mat& ridgenessImage, Mat& hessianDeterminant, Mat& hessianAngle)
	{
		// In: input image as matrix with single-channel double (0.0 to 1.0) elements
		// specularThreshold: integer range [0 255], //S* this is stupid

		// RemoveSpecular(In, specularThreshold); //S* Add later if necessary

		Mat vesselEnhancedImage = Mat1d(In.rows, In.cols);
		FrangiFilter2D(In, vesselEnhancedImage, hessianAngle, hessianDeterminant);
		int sigma = 255;
		CalculateRidgeness(In, vesselEnhancedImage, hessianAngle, sigma, ridgenessImage);

	}

	// Calculate Ridgeness
	// Inputs: In, enhancedIn
	// Outputs: 
	void CalculateRidgeness(Mat& originalImage, Mat& enhancedIn, Mat& Direction, int sigma, Mat& ridgenessImage) // Review Paramters
	{
		const int rows = originalImage.rows, cols = originalImage.cols;
		
		// Get hessianIx and hessianIy
		Mat hessianIx = Mat1d(rows, cols);
		Mat hessianIy = Mat1d(rows, cols);

		for (int i = 0; i < rows; i++)
		{
			e_t * const pHessianIx= hessianIx.ptr<e_t>(i);
			e_t * const pHessianIy = hessianIy.ptr<e_t>(i);
			e_t * const pDirection = Direction.ptr<e_t>(i);
			for (int j = 0; j < cols; j++)
			{
				e_t temp = pDirection[j] + (VESSEL_FEATURES_PI / 2);
				pHessianIx[j] = cos(temp);
				pHessianIy[j] = sin(temp);
			}
		}

		// Apply guassian filter
		const int kSize = 5;
		cv::Mat gaussianKernel = cv::getGaussianKernel(kSize, sigma, CV_64F);
		cv::mulTransposed(gaussianKernel, gaussianKernel, false);
		Mat filteredImage = Mat1d(rows, cols);
		// S* Note Below might be wrong, ImFilter implementation
		// assumes matlab parameter 'conv' not 'replicate'
		vf::Imfilter(originalImage, filteredImage, gaussianKernel);

		// Get [Y, x]
		const int kern_width = 6 * sigma + 1;
		Mat Y = Mat(kern_width, kern_width, MAT_TYPE_ID);
		for (int i = 0, val = -3 * sigma; i < kern_width; i++, val++)
		{
			e_t * const pY = Y.ptr<e_t>(i);
			for (int j = 0; j < kern_width; j++)
			{
				pY[j] = static_cast<e_t>(val);
			}
		}
		Mat X = Y.t();

		// Get DGaussx and Dgaussy
		Mat exp_term;
		cv::exp(-(X.mul(X) + Y.mul(Y)) / (2 * sigma * sigma), exp_term);
		Mat DGaussx = (1 / (2 * VESSEL_FEATURES_PI * pow(sigma, 2))) * ( (-X) / (sigma * sigma)).mul(exp_term);
		Mat DGaussy = DGaussx.t();

		// Get Dx and Dy by filtering
		Mat Dx = Mat1d(rows, cols);
		Mat Dy = Mat1d(rows, cols);
		vf::Imfilter(filteredImage, Dx, DGaussx);
		vf::Imfilter(filteredImage, Dy, DGaussy);

		//
		int margin = 3;

	}

	// Removes specular reflection from input image
	void RemoveSpecular(Mat& In, int specularThreshold = 255)
	{

	}

	// Demos the basic features provided
	void Demo(const string& testImage)
	{
		Mat image;
		image = imread(testImage, IMREAD_COLOR); // Read the file
		if (!image.data) // Check for invalid input
		{
			std::cerr << "Could not open: " << testImage << std::endl;
			exit(-1);
		}

		Mat green;
		GetGreenChannelAsDouble(image, green);
		green = green * 255.0;

		Mat vesselEnhancedImage = Mat1d(image.rows, image.cols); //* S: Change type
		Mat hessianAngle = Mat1d(image.rows, image.cols);
		Mat hessianDeterminant = Mat1d(image.rows, image.cols);

		vf::FrangiFilter2D(green, vesselEnhancedImage, hessianAngle, hessianDeterminant);

		imshow("Frangi Output Image", vesselEnhancedImage);
		imshow("Frangi Output Hessian Determinant", hessianDeterminant);
		imshow("Frangi Output Hessian Angle", hessianAngle);
	}

	// Takes an input image in which each pixel is represented a 3 bytes (RGB),
	// then outputs the green channel as a float between 0 and 1.
	void GetGreenChannelAsDouble(const Mat& In, Mat& Out)
	{
		// In: input image matrix with three byte RGB pixel format
		// Out: output image matrix with containing green channel as float between 0 and 1
		Mat channels[3];
		split(In, channels); // separate channels
		channels[1].convertTo(Out, CV_64F, 1.0 / 255.0); // covert green channel to float
	}

	void Imfilter(const Mat& src, Mat& dst, const Mat& ker)
	{
		Point anchor(-1, -1);
		double delta = 0;
		//Mat ker_rotated_180 = Mat(ker.size(), ker.type()); // Commented because there is no need to rotate kernal 180 degrees... all passed input kernals are symmetric
		//flip(ker, ker_rotated_180, -1); // -1 to flip along both axes
		//Ptr<FilterEngine> fe = createLinearFilter(src.type(), ker_rotated_180.type(), ker_rotated_180, anchor, delta, BORDER_CONSTANT, BORDER_CONSTANT, Scalar(0));
		Ptr<FilterEngine> fe = createLinearFilter(src.type(), ker.type(), ker, anchor, delta, BORDER_CONSTANT, BORDER_CONSTANT, Scalar(0));
		fe->apply(src, dst);
	}
}
