#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "watershed.h"

using namespace std;
using namespace cv;

float count_pixels(Mat res, Mat mean_shift_img) {
    int res_pixels_black = 0;
    int source_pixels_not_black = 0;

    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            if (res.at<uchar>(i, j) == 0)
                res_pixels_black += 1;
            if (mean_shift_img.at<Vec3b>(i, j) != Vec3b(0,0,0))
                if (mean_shift_img.at<Vec3b>(i, j) != Vec3b(1,1,1))
                    source_pixels_not_black += 1;
        }
    }
    float res_pixels_white = (res.rows * res.cols) - res_pixels_black;
    return  res_pixels_white / source_pixels_not_black;
}

int count_pixels_not_zero(Mat img) {
    int num = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                if (img.at<Vec3b>(i, j) != Vec3b(1, 1, 1))
                    num += 1;
        }
    }
    return num;
}


int count_pixels_with_value_n(Mat clustered, int n) {
    int count = 0;
    for (int i = 0; i < clustered.rows; i++) {
        for (int j = 0; j < clustered.cols; j++) {
            if (clustered.at<Vec3b>(i, j)[0] == n)
                count++;
        }
    }
    return count;
}

int count_cluster_pixels(Mat img) {
    int num = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) != 0)
                num += 1;
        }
    }
    return num;
}

Mat find_pasta(Mat image) {

    Scalar lowerBound = Scalar(14, 96, 70); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(24, 255, 255);   //pasta

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pesto(Mat image) {

    Scalar lowerBound = Scalar(30, 90, 40); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(42, 200, 180);   //pasta

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_tomato(Mat image) {

    Scalar lowerBound = Scalar(0, 120, 50); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(10, 255, 210);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_meat_sauce(Mat image) {

    Scalar lowerBound = Scalar(0, 70, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(11, 200, 170);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pasta_clams_mussels(Mat image) {

    Scalar lowerBound = Scalar(7, 155, 80); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(18, 255, 220);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_rice(Mat image) {

    Scalar lowerBound = Scalar(15, 82, 95); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(22, 220, 214);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_grilled_pork_cutlet(Mat image) {

    Scalar lowerBound = Scalar(9, 50, 95); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(18, 160, 200);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_fish_cutlet(Mat image) {

    Scalar lowerBound = Scalar(12, 135, 135); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(17, 220, 200);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_rabbit(Mat image) {

    Scalar lowerBound = Scalar(9, 95, 44); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(17, 255, 220);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_seafood_salad(Mat image) {

    Scalar lowerBound = Scalar(13, 28, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(21, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_beans(Mat image) {

    Scalar lowerBound = Scalar(0, 80, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(10, 180, 185);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_potatoes(Mat image) {

    Scalar lowerBound = Scalar(17, 50, 100); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(25, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}



vector<Mat> find_salad(String img_path) {
    // Read the image as rgb
    Mat img = imread(img_path);

    //we need to create a border around the image to manage circles(dishes) that exit the image (cutted):
    int width = img.cols;
    int height = img.rows;
    Mat img2(height * 2, width * 2, img.type(), Scalar(0, 0, 0));
    int startX = (img2.cols - width) / 2;
    int startY = (img2.rows - height) / 2;
    img.copyTo(img2(Rect(startX, startY, width, height)));

    // Convert to gray-scale
    Mat gray;
    cvtColor(img2, gray, COLOR_BGR2GRAY);
    // Blur the image to reduce noise 
    Mat img_blur;
    medianBlur(gray, img_blur, 5);
    // Create a vector for detected circles
    vector<Vec3f>  circles;
    // Apply Hough Transform
    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img2.rows / 2, 100, 50, 150, 200);

    vector<Mat> dishes;
    // Draw detected circles
    for (size_t i = 0; i < circles.size(); i++) {


        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img2, center, radius, Scalar(255, 255, 255), 2, 8, 0);    //draw the circle on the image
        //Extract the circles as single images:
            // Draw the mask: white circle on black background
        Mat1b mask(img2.size(), uchar(0));
        circle(mask, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255), FILLED);
        // Compute the bounding box
        Rect bbox(circles[i][0] - circles[i][2], circles[i][1] - circles[i][2], 2 * circles[i][2], 2 * circles[i][2]);
        // Create a black image
        Mat3b res(img2.size(), Vec3b(0, 0, 0));
        // Copy only the image under the white circle to black image
        img2.copyTo(res, mask);
        // Crop according to the roi
        res = res(bbox);
        // Save the image
    //String name = "C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/hough_circles/c" + to_string(i) + ".png";
    //imwrite(name, res);

        dishes.push_back(res);

    }
    return dishes;
}

vector<Mat> hough_transform(String img_path) {
    // Read the image as rgb
    Mat img = imread(img_path);

    //we need to create a border around the image to manage circles(dishes) that exit the image (cutted):
    int width = img.cols;
    int height = img.rows;
    Mat img2(height * 2, width * 2, img.type(), Scalar(0, 0, 0));
    int startX = (img2.cols - width) / 2;
    int startY = (img2.rows - height) / 2;
    img.copyTo(img2(Rect(startX, startY, width, height)));

    // Convert to gray-scale
    Mat gray;
    cvtColor(img2, gray, COLOR_BGR2GRAY);
    // Blur the image to reduce noise 
    Mat img_blur;
    medianBlur(gray, img_blur, 5);
    // Create a vector for detected circles
    vector<Vec3f>  circles;
    // Apply Hough Transform
    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img2.rows/5, 100, 100, 200, 0);

    vector<Mat> dishes;
    // Draw detected circles
    for (size_t i = 0; i < circles.size(); i++) {


        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img2, center, radius, Scalar(255, 255, 255), 2, 8, 0);    //draw the circle on the image
        //Extract the circles as single images:
            // Draw the mask: white circle on black background
        Mat1b mask(img2.size(), uchar(0));
        circle(mask, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255), FILLED);
            // Compute the bounding box
        Rect bbox(circles[i][0] - circles[i][2], circles[i][1] - circles[i][2], 2 * circles[i][2], 2 * circles[i][2]);
            // Create a black image
        Mat3b res(img2.size(), Vec3b(0, 0, 0)); 
            // Copy only the image under the white circle to black image
        img2.copyTo(res, mask);
            // Crop according to the roi
        res = res(bbox);
            // Save the image
        //String name = "C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/hough_circles/c" + to_string(i) + ".png";
        //imwrite(name, res);

        dishes.push_back(res);

    }
    return dishes;
}



Mat preprocess(Mat img) {
    //KMEANS PREPROCESSING:
    Mat bilateral_img;
    bilateralFilter(img, bilateral_img, 20, 200, 250, BORDER_DEFAULT);    //apply bilateral filter

    Mat mean_shift_img;
    pyrMeanShiftFiltering(bilateral_img, mean_shift_img, 32, 16, 3); //apply mean-shift

    //Remove the dish (white pixels):
    Scalar lowerBound = Scalar(0, 0, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255] to remove plate (bianco)
    Scalar upperBound = Scalar(255, 55, 255);
    Mat hsv;
    cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV
    Mat mask;
    inRange(hsv, lowerBound, upperBound, mask);    //filter pixels in range

    mean_shift_img.setTo(Scalar(0, 0, 0), mask);


    erode(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);   //remove small isolated parts
    dilate(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);

    return mean_shift_img;
}




vector<float> calc_mean_cluster_color(Mat hsv, Mat clusterized, int numRegions) {

    vector<float>  mean(numRegions + 1, 0);  //containing sum of H values for each label
    vector<int> count(numRegions + 1, 0);  //count number of pixels for each label to produce mean

    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            int label = clusterized.at<Vec3b>(i, j)[0];
            mean[label] += hsv.at<Vec3b>(i, j)[0];
            count[label] += 1;
        }
    }

    for (int i = 0; i < mean.size(); i++) { //calculate means
        mean[i] = mean[i] / count[i];
    }

    return mean;
}



Mat kmeans(Mat mean_shift_img, int numRegions) {


    // Create the feature matrix
    int numPixels = 0;
    for (int i = 0; i < mean_shift_img.rows; i++) {
        for (int j = 0; j < mean_shift_img.cols; j++) {
            if (mean_shift_img.at<Vec3b>(i, j)[0] < 5)  mean_shift_img.at<Vec3b>(i, j)[0] = 0;  //mean shift rende lo sfondo non nero del tutto a volte (es 1,1,1)
            if (mean_shift_img.at<Vec3b>(i, j)[1] < 5)  mean_shift_img.at<Vec3b>(i, j)[1] = 0;
            if (mean_shift_img.at<Vec3b>(i, j)[2] < 5)  mean_shift_img.at<Vec3b>(i, j)[2] = 0;

            Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
            if (pixel != Vec3b(0, 0, 0)) {
                numPixels++;
            }
        }
    }


    Mat featureMatrix(numPixels, 1, CV_32F);
    int index = 0;

    Mat hsv;
    cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV

    for (int i = 0; i < mean_shift_img.rows; i++) {
        for (int j = 0; j < mean_shift_img.cols; j++) {
            Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
            if (pixel != Vec3b(0, 0, 0)) {
                //featureMatrix.at<float>(index, 0) = static_cast<float>(pixel[0] / 255.0);  // Blue channel
                //featureMatrix.at<float>(index, 1) = static_cast<float>(pixel[1] / 255.0);  // Green channel
                //featureMatrix.at<float>(index, 2) = static_cast<float>(pixel[2] / 255.0);  // Red channel
                //featureMatrix.at<float>(index, 3) = static_cast<float>(i / mean_shift_img.rows);      // Pixel row
                //featureMatrix.at<float>(index, 4) = static_cast<float>(j / mean_shift_img.cols);      // Pixel column
                featureMatrix.at<float>(index, 0) = static_cast<float>(hsv.at<Vec3b>(i, j)[0]/* / 180.0*/);     // H hsv color
                index++;
            }
        }
    }

    // Normalize the feature matrix
    //Mat normalizedFeatures;
    //normalize(featureMatrix, normalizedFeatures, 0, 1, NORM_MINMAX);

    // Convert the reshaped image to floats
    Mat featureMatrixFloat;
    featureMatrix.convertTo(featureMatrixFloat, CV_32F);

    //compute kmeans
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2);
    kmeans(featureMatrixFloat, numRegions, labels, criteria, 10, KMEANS_PP_CENTERS, centers);
    

    //print centers matrix
    cout << "Center matrix:\n";
    for (int i = 0; i < centers.rows; i++) {
        for (int j = 0; j < centers.cols; j++) {
            cout << to_string(centers.at<float>(i, j)) << " ";
        }
        cout << "\n";
    }



    // Create an output image to display the region masks
    Mat outputImage = Mat::zeros(mean_shift_img.size(), CV_8UC3);

    // Assign pixels to their corresponding regions
    index = 0;
    for (int i = 0; i < mean_shift_img.rows; i++) {
        for (int j = 0; j < mean_shift_img.cols; j++) {
            Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
            if (pixel != Vec3b(0, 0, 0)) {
                int label = labels.at<int>(index++);
                //outputImage.at<Vec3b>(i, j) = Vec3b(centers.at<float>(label, 2), centers.at<float>(label, 1), centers.at<float>(label, 0));
                outputImage.at<Vec3b>(i, j) = Vec3b(label+1, label+1, label+1);
            }
        }
    }
   

    return outputImage;
}






int evaluate_kmeans(Mat src, Mat clusterized, int numRegions) {
    
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV

    //cout << to_string(clusterized.at<Vec3b>(1, 1)[0]) <<"  "<< to_string(clusterized.at<Vec3b>(1, 1)[1])<< "  "<< to_string(clusterized.at<Vec3b>(1, 1)[2]) << "\n";

    //label 0=background, 1,2,3 other labels
    vector<int>  mean(numRegions+1, 0);  //containing sum of H values for each label
    vector<int> count(numRegions+1, 0);  //count number of pixels for each label to produce mean

    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            int label = clusterized.at<Vec3b>(i, j)[0];
            mean[label] += hsv.at<Vec3b>(i, j)[0];
            count[label] += 1;
        }
    }

    for (int i = 0; i < mean.size(); i++) { //calculate means
        mean[i] = mean[i] / count[i];
    }

    int total_error = 0;

    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            int label = clusterized.at<Vec3b>(i, j)[0];
            if (label != 0) 
                total_error += abs(hsv.at<Vec3b>(i, j)[0] - mean[label]);
        }
    }

    return total_error;
}



int findBestNumClusters(Mat mean_shift_img) {
    vector<double> distortion;
    for (int k = 1; k <= 3; k++) {  // Adjust the loop to search only for k values of 1, 2, and 3
        
        // Create the feature matrix
        int numPixels = 0;
        for (int i = 0; i < mean_shift_img.rows; i++) {
            for (int j = 0; j < mean_shift_img.cols; j++) {
                if (mean_shift_img.at<Vec3b>(i, j)[0] < 5)  mean_shift_img.at<Vec3b>(i, j)[0] = 0;  //mean shift rende lo sfondo non nero del tutto a volte (es 1,1,1)
                if (mean_shift_img.at<Vec3b>(i, j)[1] < 5)  mean_shift_img.at<Vec3b>(i, j)[1] = 0;
                if (mean_shift_img.at<Vec3b>(i, j)[2] < 5)  mean_shift_img.at<Vec3b>(i, j)[2] = 0;

                Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
                if (pixel != Vec3b(0, 0, 0)) {
                    numPixels++;
                }
            }
        }

        Mat featureMatrix(numPixels, 1, CV_32F);
        int index = 0;

        Mat hsv;
        cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV

        for (int i = 0; i < mean_shift_img.rows; i++) {
            for (int j = 0; j < mean_shift_img.cols; j++) {
                Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
                if (pixel != Vec3b(0, 0, 0)) {
                    featureMatrix.at<float>(index, 0) = static_cast<float>(hsv.at<Vec3b>(i, j)[0]/* / 180.0*/);     // H hsv color
                    index++;
                }
            }
        }

        // Convert the reshaped image to floats
        Mat featureMatrixFloat;
        featureMatrix.convertTo(featureMatrixFloat, CV_32F);

        //compute kmeans
        Mat labels, centers;
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2);
        kmeans(featureMatrixFloat, k, labels, criteria, 10, KMEANS_PP_CENTERS, centers);

        //compute distance
        double sumDist = 0.0;
        for (int i = 0; i < featureMatrixFloat.rows; i++) {
            int label = labels.at<int>(i);
            Vec2f point(featureMatrixFloat.at<float>(i, 0));    // only one dimension of features size
            Vec2f center(centers.at<float>(label, 0));
            sumDist += norm(point, center);
        }

        double avgDist = sumDist / featureMatrixFloat.rows;
        distortion.push_back(avgDist);

    }

    // Find the best number of clusters
    int bestNumClusters = 1;
    double minDistortion = distortion[0];
    for (int i = 1; i < distortion.size(); i++) {
        if (distortion[i] < minDistortion) {  // Adjust the comparison to find the minimum distortion
            bestNumClusters = i + 1;
            minDistortion = distortion[i];
        }
    }


    return bestNumClusters;
}



Mat print_clustered_img(Mat img) {

    Mat outputImage = Mat::zeros(img.size(), CV_8UC3);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            outputImage.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i, j)[0]) * 50;
            outputImage.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i, j)[1]) * 50;
            outputImage.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i, j)[2]) * 50;
        }
    }
    return outputImage;
}






//BAG OF WORDS
/*
* 
* // Function to compute descriptors for all images in a given folder
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




<
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


    cout << "Histogram for the test image:" << endl;
    for (int i = 0; i < bowDescriptor.cols; ++i) {
        cout << "Column " << i << ": " << bowDescriptor.at<float>(0, i) << endl;
    }

    return 0;
*/