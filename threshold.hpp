#ifndef THRESHOLD_HPP_
#define THRESHOLD_HPP_

#include <vector>
#include <tuple>

std::vector<std::tuple<int,int,int> > get_image(int &width, int &height);

std::vector<int> convert_to_greyscale(std::vector<std::tuple<int,int,int> > image, int width, int height);

std::vector<int> create_histogram(std::vector<int> greyscale, int width, int height);

std::vector<double> memoize_P(std::vector<int> histogram, int width, int height);

double variance(std::vector<double> P);

int automatic_threshold(std::vector<int> greyscale, int width, int height);

#endif // THRESHOLD_HPP_
