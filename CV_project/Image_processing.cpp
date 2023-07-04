#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "watershed.h"

using namespace std;
using namespace cv;


Mat find_pasta(Mat image) {

    Scalar lowerBound = Scalar(11, 80, 65); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(23, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pesto(Mat image) {
    
    Scalar lowerBound = Scalar(30, 60, 30); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(45, 255, 255);
    
    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pomodoro(Mat image) {

    Scalar lowerBound = Scalar(0, 190, 45); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(12, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pasta_cozze(Mat image) {

    Scalar lowerBound = Scalar(9, 148, 69); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(18, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_ragu(Mat image) {

    Scalar lowerBound = Scalar(5, 85, 50); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(10, 255, 200);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_fagioli(Mat image) {

    Scalar lowerBound = Scalar(6, 83, 40); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(9, 255, 180);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_carne(Mat image) {

    Scalar lowerBound = Scalar(9, 50, 70); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(14, 160, 200);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_patate(Mat image) {

    Scalar lowerBound = Scalar(17, 45, 100); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(23, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
}

Mat find_pesce(Mat image) {

    Scalar lowerBound = Scalar(12, 125, 125); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(15, 200, 200);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range
    return output_mask;
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
    for (int i = 0; i < centers.rows; i++) {
        for (int j = 0; j < centers.cols; j++) {
            cout << to_string(centers.at<float>(i, j)) << "\n";
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
    imshow("cluster", outputImage);

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
            if(label != 0)
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