#pragma once
#include <opencv2\opencv.hpp>

std::vector<cv::Mat> hough_transform(cv::String img_path);

bool find_pasta_pesto(int c1, int c2);
bool find_pasta_pomodoro(int c1, int c2);
bool find_pasta_cozze(int c1);
bool find_pasta_ragu(int c1, int c2);
cv::Mat find_fagioli(cv::Mat image);
cv::Mat find_carne(cv::Mat image);
cv::Mat find_patate(cv::Mat image);
cv::Mat find_pesce(cv::Mat image);


cv::Mat kmeans(cv::Mat image, int numRegions);
int evaluate_kmeans(cv::Mat src, cv::Mat clusterized, int numRegions);
cv::Mat print_clustered_img(cv::Mat img);

int findBestNumClusters(cv::Mat mean_shift_img);

//////////////////////
double calculateSilhouetteScore(const cv::Mat& labels, const cv::Mat& distances, int k);