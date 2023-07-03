#pragma once
#include <opencv2\opencv.hpp>

void hough_transform(cv::String img_path);

cv::Mat find_pasta(cv::Mat image);
cv::Mat find_pesto(cv::Mat image);
cv::Mat find_pomodoro(cv::Mat image);
cv::Mat find_pasta_cozze(cv::Mat image);
cv::Mat find_ragu(cv::Mat image);
cv::Mat find_fagioli(cv::Mat image);
cv::Mat find_carne(cv::Mat image);
cv::Mat find_patate(cv::Mat image);
cv::Mat find_pesce(cv::Mat image);

void calculate_food(cv::Mat image);

cv::Mat kmeans(cv::Mat image, int numRegions);
int evaluate_kmeans(cv::Mat src, cv::Mat clusterized, int numRegions);
void print_clustered_img(cv::Mat img);