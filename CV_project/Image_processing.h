#pragma once
#include <opencv2\opencv.hpp>

std::vector<cv::Mat> hough_transform(cv::Mat img);
std::vector<cv::Mat> find_salad(cv::Mat img);
int find_remaining_salad(cv::Mat image);

cv::Point food_recognition_rectangle(cv::Mat source, cv::Mat template_img);

std::vector<cv::Point> calculate_template_from_cluster(cv::Mat source, cv::Mat clustered, cv::Mat dish, int label);

float count_pixels(cv::Mat img, cv::Mat mean_shift_img);
int count_pixels_not_zero(cv::Mat img);
int count_cluster_pixels(cv::Mat img);

cv::Mat preprocess(cv::Mat img);

std::vector<float> calc_mean_cluster_color(cv::Mat hsv, cv::Mat clusterized, int numRegions);

int count_pixels_with_value_n(cv::Mat clustered, int n);

cv::Mat find_pasta(cv::Mat image);
cv::Mat find_pesto(cv::Mat image);
cv::Mat find_tomato(cv::Mat image);
cv::Mat find_meat_sauce(cv::Mat image);
cv::Mat find_pasta_clams_mussels(cv::Mat image);
cv::Mat find_rice(cv::Mat image);
cv::Mat find_grilled_pork_cutlet(cv::Mat image);
cv::Mat find_fish_cutlet(cv::Mat image);
cv::Mat find_rabbit(cv::Mat image);
cv::Mat find_seafood_salad(cv::Mat image);
cv::Mat find_beans(cv::Mat image);
cv::Mat find_potatoes(cv::Mat image);


cv::Mat kmeans(cv::Mat image, int numRegions);

cv::Mat print_clustered_img(cv::Mat img);


