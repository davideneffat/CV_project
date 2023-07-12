#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

/*void trainClassifier(const string& trainingFolder, Ptr<Feature2D>& featureExtractor, Ptr<BOWKMeansTrainer>& bowTrainer, Ptr<SVM>& svm)
{
    // Load training images from the specified folder
    vector<string> imagePaths;
    glob(trainingFolder, imagePaths);

    // Extract features from training images
    Mat descriptors;
    for (const auto& imagePath : imagePaths)
    {
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        vector<KeyPoint> keypoints;
        featureExtractor->detect(image, keypoints);
        Mat descriptor;
        featureExtractor->compute(image, keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // Cluster features using k-means
    Mat vocabulary = bowTrainer->cluster(descriptors);

    // Train SVM classifier
    Mat labels(imagePaths.size(), 1, CV_32SC1);
    for (int i = 0; i < imagePaths.size(); i++)
    {
        labels.at<int>(i) = i / 10; // Assuming 10 images per class
    }
    Mat trainingData;
    bowTrainer->setVocabulary(vocabulary);
    for (const auto& imagePath : imagePaths)
    {
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        vector<KeyPoint> keypoints;
        featureExtractor->detect(image, keypoints);
        Mat descriptor;
        featureExtractor->compute(image, keypoints, descriptor);
        Mat bowDescriptor;
        bowTrainer->compute(descriptor, bowDescriptor);
        trainingData.push_back(bowDescriptor);
    }
    svm->train(trainingData, ROW_SAMPLE, labels);
}

void testClassifier(const string& testFolder, Ptr<Feature2D>& featureExtractor, Ptr<BOWImgDescriptorExtractor>& bowExtractor, Ptr<SVM>& svm)
{
    // Load test images from the specified folder
    vector<string> imagePaths;
    glob(testFolder, imagePaths);

    // Classify test images
    for (const auto& imagePath : imagePaths)
    {
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);
        vector<KeyPoint> keypoints;
        featureExtractor->detect(image, keypoints);
        Mat descriptor;
        featureExtractor->compute(image, keypoints, descriptor);
        Mat bowDescriptor;
        bowExtractor->compute(descriptor, bowDescriptor);
        int predictedClass = svm->predict(bowDescriptor);
        cout << "Image: " << imagePath << ", Predicted Class: " << predictedClass << endl;
    }
}*/