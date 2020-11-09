#include "opencv2\opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "image_compression.h"

int main(int argc, char* argv[]) {
	int k = 64;
	int iters = 300;
	std::string imageName = "F:/CUDA/image_compression/images/inputDoor.png";
	cv::Mat image = imread(imageName, cv::IMREAD_COLOR);
	if (image.empty()) {
		std::cout << "Invalid image. Please try again" << std::endl;
		return 1;
	}
	imshow("image", image);

	std::cout << "Image Compression, k = " << k << std::endl;
	cv::Mat result = kmeans_cuda(image, k, iters);
	cv::imshow("result", result);

	cv::waitKey();
	return 0;
}