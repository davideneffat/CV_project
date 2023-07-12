
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "Image_processing.h"
#include "watershed.h"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>




using namespace std;
using namespace cv;
using namespace cv::ml;


const int numClasses = 1;
const int numClusters = 100;
const string trainingDataPath = "Food_leftover_dataset/dataset/fagioli/imgs/";  //training data
const string vocabularyFile = "vocabulary.xml";
const string svmModelFile = "svm_model.xml";

void extractFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors)
{
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    fast->detect(image, keypoints);
    //sift->compute(image, keypoints, descriptors);
    cout << "extract sift:\n";
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    // Check if feature descriptors are empty
    if (descriptors.empty())
    {
        std::cout << "No feature descriptors found." << std::endl;
    }
    cout << to_string(descriptors.cols) << "\n";
}

void buildVocabulary(const vector<Mat>& trainingImages, Mat& vocabulary)
{
    Ptr<BOWKMeansTrainer> bowTrainer = makePtr<BOWKMeansTrainer>(numClusters);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    
    for (const auto& image : trainingImages)
    {
        cout << "a\n";
        vector<KeyPoint> keypoints;
        Mat descriptors;
        extractFeatures(image, keypoints, descriptors);
        cout << "b\n";
        bowTrainer->add(descriptors);
        cout << "c\n";
    }
    
    vocabulary = bowTrainer->cluster();
    cout << "Clustering ended\n";
}

void computeBowDescriptor(const Mat& image, const Mat& vocabulary, Mat& bowDescriptor)
{
    Ptr<FeatureDetector> orb = ORB::create();
    Ptr<DescriptorExtractor> orbDescriptorExtractor = ORB::create();

    vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detect(image, keypoints);
    orbDescriptorExtractor->compute(image, keypoints, descriptors);

    // Create a BOWImgDescriptorExtractor with a BruteForce matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowExtractor = makePtr<BOWImgDescriptorExtractor>(orbDescriptorExtractor, matcher);
    bowExtractor->setVocabulary(vocabulary);

    bowExtractor->compute(image, keypoints, bowDescriptor);
}



void trainClassifier(const vector<Mat>& trainingImages, const Mat& vocabulary, const vector<int>& labels, Ptr<cv::ml::SVM>& svm)
{
    Mat trainingData;
    vector<int> trainingLabels;

    for (size_t i = 0; i < trainingImages.size(); i++)
    {
        Mat bowDescriptor;
        computeBowDescriptor(trainingImages[i], vocabulary, bowDescriptor);
        trainingData.push_back(bowDescriptor);
        trainingLabels.push_back(labels[i]);
    }

    Mat trainingDataFloat;
    trainingData.convertTo(trainingDataFloat, CV_32FC1);

    svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->train(trainingDataFloat, ROW_SAMPLE, Mat(trainingLabels));
    svm->save(svmModelFile);
}

int classifyImage(const Mat& image, const Mat& vocabulary, Ptr<cv::ml::SVM>& svm)
{
    Mat bowDescriptor;
    computeBowDescriptor(image, vocabulary, bowDescriptor);
    Mat bowDescriptorFloat;
    bowDescriptor.convertTo(bowDescriptorFloat, CV_32FC1);

    return svm->predict(bowDescriptorFloat);
}



int main(int argc, char* argv[])
{
    // Step 1: Prepare the Training Data
    vector<Mat> trainingImages;
    vector<int> labels;
    
    for (int i = 0; i < numClasses; i++)
    {
        for (int j=0; j<5; j++)   //20 images training set
        {
            string imagePath = trainingDataPath + "fagioli" + to_string(j) + ".jpg";
            Mat image = imread(imagePath, IMREAD_GRAYSCALE);
            trainingImages.push_back(image);
            labels.push_back(i + 1);
        }
    }
    cout << "AAAAAAAAAAAAAAA\n";
    // Step 2: Feature Extraction using FAST

    // Step 3: Build the Visual Vocabulary (Bag of Words)
    Mat vocabulary;
    buildVocabulary(trainingImages, vocabulary);
    cout << "11111111111111\n";
    FileStorage fs(vocabularyFile, FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    cout << "BBBBBBBBBBBBBBB\n";
    // Step 4: Train the Classifier
    Ptr<cv::ml::SVM> svm;
    trainClassifier(trainingImages, vocabulary, labels, svm);
    cout << "CCCCCCCCCCCCCCCCCCC\n";
    // Step 5: Classify New Images
    Mat testImage = imread("Food_leftover_dataset/dataset/fagioli/imgs/fagioli20.jpg", IMREAD_GRAYSCALE);
    int predictedClass = classifyImage(testImage, vocabulary, svm);
    cout << "Predicted class: " << to_string(predictedClass) << endl;






    // Load the two input images
    /*cv::Mat image1 = cv::imread("Food_leftover_dataset/tray1/food_image.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread("Food_leftover_dataset/tray1/leftover1.jpg", cv::IMREAD_GRAYSCALE);
    imshow("A", image1);
    imshow("B", image2);


    // Convert the images to 8-bit grayscale if needed
    if (image1.type() != CV_8U || image2.type() != CV_8U)
    {
        cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
    }

    // Step 1: Detect features using FAST
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create();
    fast->detect(image1, keypoints1);
    fast->detect(image2, keypoints2);

    // Step 2: Extract feature descriptors using SIFT
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Check if feature descriptors are empty
    if (descriptors1.empty() || descriptors2.empty())
    {
        std::cout << "No feature descriptors found." << std::endl;
        return -1;
    }

    // Step 3: Match feature descriptors using BruteForce
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Apply ratio test to filter good matches
    const float ratioThreshold = 0.7f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& matches : knnMatches)
    {
        if (matches.size() >= 2 && matches[0].distance < ratioThreshold * matches[1].distance)
        {
            goodMatches.push_back(matches[0]);
        }
    }

    // Step 4: Draw the matches
    cv::Mat matchedImage;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, goodMatches, matchedImage);

    // Display the matched image
    cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    cv::imshow("Matches", matchedImage);*/
    




    vector<Mat> dishes = hough_transform("Food_leftover_dataset/start_imgs/imgs/im1.jpg");
    cout << "Finded " << to_string(dishes.size()) << " dishes\n";



    for (int i = 0; i < dishes.size(); i++) {

        Mat src = dishes[i];
        string name = "input src" + to_string(i);
        imshow(name, src);



        /*
        Mat vassoio = imread("Food_leftover_dataset/start_imgs/imgs/im1.jpg");
        Mat result;
        matchTemplate(vassoio, src, result, TM_SQDIFF);
        normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

        matchLoc = minLoc;

        Mat img_display;
        vassoio.copyTo(img_display);
        rectangle(img_display, matchLoc, Point(matchLoc.x + src.cols, matchLoc.y + src.rows), Scalar::all(0), 2, 8, 0);
        rectangle(result, matchLoc, Point(matchLoc.x + src.cols, matchLoc.y + src.rows), Scalar::all(0), 2, 8, 0);
        name = "image_window" + to_string(i);
        imshow(name, img_display);
        name = "result_window" + to_string(i);
        imshow(name, result);

        //KMEANS PREPROCESSING:
        Mat bilateral_img;
        bilateralFilter(src, bilateral_img, 20, 200, 250, BORDER_DEFAULT);    //apply bilateral filter
        name = "Bilateral filter" + to_string(i);
        imshow(name, bilateral_img);

        Mat mean_shift_img;
        pyrMeanShiftFiltering(bilateral_img, mean_shift_img, 32, 16, 3); //apply mean-shift
        //pyrMeanShiftFiltering(bilateral_img, mean_shift_img, 20, 45, 3); //apply mean-shift
        name = "Mean-shift" + to_string(i);
        imshow(name, mean_shift_img);

        //Remove the dish (white pixels):
        Scalar lowerBound = Scalar(0, 0, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255] to remove plate (bianco)
        Scalar upperBound = Scalar(255, 55, 255);
        Mat hsv;
        cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV
        Mat mask;
        inRange(hsv, lowerBound, upperBound, mask);    //filter pixels in range

        mean_shift_img.setTo(Scalar(0, 0, 0), mask);
        name = "plate removal" + to_string(i);
        imshow(name, mean_shift_img);


        erode(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);   //remove small isolated parts
        dilate(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);
        name = "erosion+dilation" + to_string(i);
        imshow(name, mean_shift_img);


        //apply kmeans with k=1,2,3 and find best value for k
        Mat res1 = kmeans(mean_shift_img, 1);
        Mat out1 = print_clustered_img(res1);
        name = "k=1_" + to_string(i);
        imshow(name, out1);
        int err1 = evaluate_kmeans(mean_shift_img, res1, 1);
        cout << "Errore con 1 cluster = " << to_string(err1) << "\n";

        Mat res2 = kmeans(mean_shift_img, 2);
        Mat out2 = print_clustered_img(res2);
        name = "k=2_" + to_string(i);
        imshow(name, out2);
        int err2 = evaluate_kmeans(mean_shift_img, res2, 2);
        //err2 = err2 * 2;
        cout << "Errore con 2 cluster = " << to_string(err2) << "\n";

        Mat res3 = kmeans(mean_shift_img, 3);
        Mat out3 = print_clustered_img(res3);
        name = "k=3_" + to_string(i);
        imshow(name, out3);
        int err3 = evaluate_kmeans(mean_shift_img, res3, 3);
        //err3 = err3 * 3;
        cout << "Errore con 3 cluster = " << to_string(err3) << "\n";

        int best_k; //number of clusters that minimizes variance
        if ((err1 <= err2) && (err1 <= err3)) {
            best_k = 1;
        }
        else if ((err2 <= err1) && (err2 <= err3))
            best_k = 2;
        else if ((err3 <= err1) && (err3 <= err2))
            best_k = 3;

        cout << to_string(best_k) << "\n";

        cout << to_string(findBestNumClusters(src)) << "\n";




        //FIND BEST K CORRECTLY
        Mat clustered = kmeans(mean_shift_img, best_k);
        Mat clustered_print = print_clustered_img(clustered);
        name = "Best_clustering_" + to_string(i);
        imshow(name, clustered_print);

        vector<float>  mean(best_k, 0);   //contains mean H (hsv) value for each cluster, mean[0]=mean val for cluster 1, mean[1]=mean val for cluster 2...
        vector<int> count(best_k, 0);  //count number of pixels for each label to produce mean

        Mat src_hsv;    //we use hsv image of src, not of mean_shift as before
        cvtColor(src, src_hsv, COLOR_BGR2HSV);  //transform from RGB to HSV

        for (int i = 0; i < clustered.rows; i++) {
            for (int j = 0; j < clustered.cols; j++) {
                int label = clustered.at<Vec3b>(i, j)[0];
                if (label != 0) {   //don't take into account background (label = 0)
                    mean[label - 1] += src_hsv.at<Vec3b>(i, j)[0];  //take value H of hsv image
                    count[label - 1] += 1;
                }
            }
        }

        for (int i = 0; i < mean.size(); i++) { //calculate means
            mean[i] = mean[i] / count[i];
        }

        colors.push_back(mean);

        //dilate(res3, res3, Mat(), Point(-1, -1), 2, 1, 1);
        //erode(res3, res3, Mat(), Point(-1, -1), 2, 1, 1);
        //Mat out3_dilated = print_clustered_img(res3);
        //imshow("res3_dilated", out3_dilated);
        */

    }


    /*
    vector<int> foods_founded(13);
    bool first_course_founded = false;

    //Find first course
    for (int i = 0; i < colors.size(); i++) {
        if (colors[i].size() == 2) {    //dish has 2 clusters (1 for pasta and 1 for pesto)
            if (find_pasta_pesto(colors[i][0], colors[i][1])) {
                foods_founded[1] = 1;
                first_course_founded = true;    //we can also delete the dish from the vector
                break;
            }
            if (find_pasta_pomodoro(colors[i][0], colors[i][1])) {
                foods_founded[2] = 1;
                first_course_founded = true;
                break;
            }
            if (find_pasta_ragu(colors[i][0], colors[i][1])) {
                foods_founded[3] = 1;
                first_course_founded = true;
                break;
            }
        }
        else if (colors[i].size() == 1) {
            if (find_pasta_cozze(colors[i][0])) {
                foods_founded[4] = 1;
                first_course_founded = true;
                break;
            }
        }
    }

    //Find second course

    for (int i = 0; i < foods_founded.size(); i++) {
        cout << to_string(foods_founded[i]) << "\n";
    }

    for (int i = 0; i < colors.size(); i++) {
        for (int j = 0; j < colors[i].size(); j++) {
            cout << to_string(colors[i][j]) << "  ";
        }
        cout << "\n";
    }
    */

    waitKey(0);
}