
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


// Function to compute descriptors for all images in a given folder
void computeDescriptors(const string& folder, vector<Mat>& descriptors, Ptr<Feature2D>& detector)
{
    vector<String> filenames;
    glob(folder, filenames);

    for (const auto& filename : filenames) {
        Mat image = imread(filename, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Failed to read image: " << filename << endl;
            continue;
        }

        vector<KeyPoint> keypoints;
        detector->detect(image, keypoints);

        Mat descriptor;
        detector->compute(image, keypoints, descriptor);

        descriptors.push_back(descriptor);
    }
}

int main()
{
    const string trainingFolder = "Food_leftover_dataset/dataset/BoW/";
    const string testImage = "Food_leftover_dataset/start_imgs/hough_circles/1.jpg";
    const int dictionarySize = 100; // Number of visual words in the vocabulary

    // Create SIFT detector
    Ptr<Feature2D> detector = SIFT::create();

    // Compute descriptors for training images
    vector<Mat> trainingDescriptors;
    computeDescriptors(trainingFolder, trainingDescriptors, detector);

    // Create BOWKMeansTrainer
    Ptr<BOWKMeansTrainer> bowTrainer = makePtr<BOWKMeansTrainer>(dictionarySize);

    // Add training descriptors to BOWTrainer
    for (const auto& descriptor : trainingDescriptors) {
        bowTrainer->add(descriptor);
    }

    // Create vocabulary (dictionary) using clustering
    Mat vocabulary = bowTrainer->cluster();

    // Create BOW descriptor extractor
    Ptr<DescriptorMatcher> descriptorMatcher = BFMatcher::create(NORM_L2);
    Ptr<BOWImgDescriptorExtractor> bowExtractor = makePtr<BOWImgDescriptorExtractor>(detector, descriptorMatcher);
    bowExtractor->setVocabulary(vocabulary);

    // Prepare training data and labels
    Mat trainingData;
    Mat labels;

    int count = 0;
    int class_count = 0;
    for (const auto& descriptor : trainingDescriptors) {
        Mat bowDescriptor;
        bowExtractor->compute(descriptor, bowDescriptor);
        trainingData.push_back(bowDescriptor);
        // Assign label based on the class (e.g., 0 for class 1, 1 for class 2, etc.)
        labels.push_back(class_count); // Update with appropriate labels for your dataset
        count++;
        if (count % 4 == 0)     //Each class have 4 images
            class_count++;
    }

    // Train the classifier (e.g., using SVM)
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::RBF);
    svm->train(trainingData, ml::ROW_SAMPLE, labels);

    // Load test image
    Mat testImageGray = imread(testImage, IMREAD_GRAYSCALE);
    if (testImageGray.empty()) {
        cerr << "Failed to read test image: " << testImage << endl;
        return -1;
    }

    // Compute descriptors for the test image
    vector<KeyPoint> keypoints;
    detector->detect(testImageGray, keypoints);

    Mat testDescriptor;
    detector->compute(testImageGray, keypoints, testDescriptor);

    // Compute the BOW descriptor for the test image
    Mat bowDescriptor;
    bowExtractor->compute(testDescriptor, bowDescriptor);

    // Predict the class of the test image
    float response = svm->predict(bowDescriptor);

    cout << "Predicted class: " << response << endl;


    // Class probabilities
    // Predict the class of the test image
    float decisionValue = svm->predict(bowDescriptor, noArray(), ml::StatModel::Flags::RAW_OUTPUT);

    // Compute class probabilities using Platt scaling
    double A = 1.0;  // A parameter for Platt scaling
    double B = 0.0;  // B parameter for Platt scaling
    double probPositive = 1.0 / (1.0 + exp(A * decisionValue + B));
    double probNegative = 1.0 - probPositive;

    cout << "Predicted class: " << response << endl;
    cout << "Class 0 Probability: " << probNegative << endl;
    cout << "Class 1 Probability: " << probPositive << endl;



    return 0;






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