#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "vessel_features.hpp"

using namespace cv;
using namespace std;

void tellMat(const Mat& m)
{
	cout << "MATRIX: size: " << m.size();
	cout << ", channels: " << m.channels();
	cout << ", elemSize: " << m.elemSize();
	cout << ", type: " << m.type();
	cout << ", depth: " << m.depth() << endl << endl;
}

void tempTest()
{
	//
	RidgeFeature x;
	Mat a, b, c, d, e;
	x.FrangiFilter2D(a, b, c, d);
	cout << "atan2(3.4, 4.7)" << atan2(3.4, 4.7) << endl;
	cout << "pow(2.0, 3)" << pow(2.0, 3) << endl;

	


	//

	Mat image;
	image = imread("hamlyn.bmp", IMREAD_COLOR); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		exit(-1);
	}


	cout << "image" << endl;
	tellMat(image);


	cout << "CV_8U: " << CV_8U << endl;
	cout << "CV_8UC3: " << CV_8UC3 << endl;
	cout << "CV_64F: " << CV_64F << endl;
	cout << "CV_64FC1: " << CV_64FC1 << endl;
	cout << "CV_64F3: " << CV_64FC3 << endl;

	image.convertTo(image, CV_64F, 1.0 / 255.0);

	Mat channels[3];
	split(image, channels);

	Mat g = channels[1];

	cout << "g" << endl;
	tellMat(g);

	Mat Ixx, Iyy, Ixy;
	RidgeFeature rf;
	rf.Hessian2D(g, 4, Ixx, Iyy, Ixy);

	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image); // Show our image inside it.

	namedWindow("Green window", WINDOW_AUTOSIZE);
	imshow("Green window", channels[1]);

	namedWindow("Green64F window", WINDOW_AUTOSIZE);
	imshow("Green64F window", channels[1]);

	namedWindow("Ixx", WINDOW_AUTOSIZE);
	imshow("Ixx", Ixx);

	namedWindow("Iyy", WINDOW_AUTOSIZE);
	imshow("Iyy", Iyy);

	namedWindow("Ixy", WINDOW_AUTOSIZE);
	imshow("Ixy", Ixy);
}

int main(int argc, char** argv)
{
	RidgeFeature::Demo();
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}