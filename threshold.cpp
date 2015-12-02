#include <iostream>
#include <vector>
#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 

#include "threshold.hpp"

// N is just the maximum value a pixel can take
#define N 255

#define index_image(image, x, y) (image)[(y)*(width)+(x)]

using namespace cv;

std::vector<std::tuple<int,int,int> > get_image(int &width, int &height) {
	std::vector<std::tuple<int,int,int> >image;
	Mat cv_img;
	cv_img = imread("image.png", 1);
	width = cv_img.cols;
	height = cv_img.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Vec3b pix = cv_img.at<Vec3b>(y,x);
			// pix is BRG and we want RGB so 1 2 0
			image.push_back(std::make_tuple(pix[1], pix[2], pix[0]));
		}
	}

	return image;
}

std::vector<int> convert_to_greyscale(std::vector<std::tuple<int,int,int> > image, int width, int height) {
	// There are many ways to convert to 'greyscale', since it isn't unique. Here we use a very simple one.
	std::vector<int> greyscale(width*height, 0);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			std::tuple<int,int,int> pix = index_image(image, x, y);
			int r = std::get<0>(pix);
			int g = std::get<1>(pix);
			int b = std::get<2>(pix);
			index_image(greyscale, x, y) = 0.2126 * r + 0.7152 * g + 0.0722 * b;
		}
	}

	// The following part is required because otherwise P[0] becomes 0
	// and we get division by zero errors when computing the between-group variance.
	int min_val = 255;
	for (int v: greyscale) {
		if (v < min_val) {
			min_val = v;
		}
	}
	for (int &v: greyscale) {
		v -= min_val;
	}

	return greyscale;
}

std::vector<int> create_histogram(std::vector<int> greyscale, int width, int height) {
	std::vector<int> histogram(N, 0);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[index_image(greyscale,x,y)]+=1;
		}
	}
	return histogram;
}	

std::vector<double> memoize_P(std::vector<int> histogram, int width, int height) {
	std::vector<double> P(N, 0);
	int size = width*height;
	for (int z = 0; z < N; z++) {
		P[z] = ((double)histogram[z])/size;
	}
	return P;
}

double variance(std::vector<double> P) {
	double mu = 0.0;
	for (int z = 0; z < N; z++) {
		mu += z*P[z];
	}
	double total_variance = 0.0;
	for (int z = 0; z < N; z++) {
		total_variance += (z - mu)*(z - mu) * P[z];
	}
	return total_variance;
}

int automatic_threshold(std::vector<int> greyscale, int width, int height) {
	std::vector<int> histogram = create_histogram(greyscale, width, height);

	std::vector<double> P = memoize_P(histogram, width, height);
	//double total_variance = variance(P);
	std::vector<double> between_variance(N, 0.0);

	std::vector<double> q_0(N, 0.0);
	std::vector<double> mu_0(N, 0.0);
	std::vector<double> mu_1(N, 0.0);

	double mu = 0.0;
	for (int z = 0; z < N; z++) {
		mu += z*P[z];
	}

	q_0[0] = P[0];

	mu_0[0] = 0.0;
	mu_1[0] = mu/(1 - q_0[0]); // since mu = q_1*mu_0 + q_1*mu_1, mu_0[0] = 0, and q_1 = 1-q_0

	between_variance[0] = q_0[0]*(1 - q_0[0])*(mu_0[0] - mu_1[0])*(mu_0[0] - mu_1[0]);
	for (int z = 1; z < N; z++) {
		q_0[z] = P[z] + q_0[z-1];
		mu_0[z] = (z * P[z])/q_0[z] + (mu_0[z-1]*q_0[z-1])/q_0[z];
		mu_1[z] = (mu - q_0[z]*mu_0[z])/(1 - q_0[z]);
		between_variance[z] = q_0[z]*(1 - q_0[z])*(mu_0[z] - mu_1[z])*(mu_0[z] - mu_1[z]);
	}

	double max_between_variance = between_variance[0];
	int max_threshold = 0;
	for (int z = 0; z < N; z++) {
		double b = between_variance[z];
		if (b > max_between_variance) {
			max_threshold = z;
			max_between_variance = b;
		}
	}

	return max_threshold;
}

int main() {
	int width, height;
	std::vector<std::tuple<int,int,int> > image = get_image(width, height);
	std::vector<int> greyscale = convert_to_greyscale(image, width, height);
	int threshold = automatic_threshold(greyscale, width, height);
	std::cout << "Threshold value: " << threshold << std::endl;

	Mat output(height, width, CV_8U);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (index_image(greyscale, x, y) <= threshold) {
				output.at<uchar>(y, x) = 255;
			}
			else {
				output.at<uchar>(y, x) = 0;
			}
		}
	}
	imwrite("output.png", output);
}
