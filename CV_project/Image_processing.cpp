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

void calculate_food(Mat image) {
    int image_pixels = image.rows * image.cols;

    int pasta_pixels = countNonZero(find_pasta(image));
    cout << "\n Area pasta = " << to_string(((double)pasta_pixels / image_pixels) * 100.0) << "%";
    int pesto_pixels = countNonZero(find_pesto(image));
    cout << "\n Area pesto = " << to_string(((double)pesto_pixels / image_pixels) * 100.0) << "%";
    int pomodoro_pixels = countNonZero(find_pomodoro(image));
    cout << "\n Area pomodoro = " << to_string(((double)pomodoro_pixels / image_pixels) * 100.0) << "%";
    int pasta_cozze_pixels = countNonZero(find_pasta_cozze(image));
    cout << "\n Area pasta cozze = " << to_string(((double)pasta_cozze_pixels / image_pixels) * 100.0) << "%";
    int ragu_pixels = countNonZero(find_ragu(image));
    cout << "\n Area ragu = " << to_string(((double)ragu_pixels / image_pixels) * 100.0) << "%";
    int fagioli_pixels = countNonZero(find_fagioli(image));
    cout << "\n Area fagioli = " << to_string(((double)fagioli_pixels / image_pixels) * 100.0) << "%";
    int carne_pixels = countNonZero(find_carne(image));
    cout << "\n Area carne = " << to_string(((double)carne_pixels / image_pixels) * 100.0) << "%";
    int patate_pixels = countNonZero(find_patate(image));
    cout << "\n Area patate = " << to_string(((double)patate_pixels / image_pixels) * 100.0) << "%";
    int pesce_pixels = countNonZero(find_pesce(image));
    cout << "\n Area pesce = " << to_string(((double)pesce_pixels / image_pixels) * 100.0) << "%";
}

void hough_transform(String img_path) {
    // Read the image as rgb
    Mat img = imread(img_path);
    // Convert to gray-scale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    // Blur the image to reduce noise 
    Mat img_blur;
    medianBlur(gray, img_blur, 5);
    // Create a vector for detected circles
    vector<Vec3f>  circles;
    // Apply Hough Transform
    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img.rows/5, 100, 100, 200, 0);
    // Draw detected circles
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(img, center, radius, Scalar(255, 255, 255), 2, 8, 0);    //draw the circle on the image
        //Extract the circles as single images:
            // Draw the mask: white circle on black background
        Mat1b mask(img.size(), uchar(0));
        circle(mask, Point(circles[i][0], circles[i][1]), circles[i][2], Scalar(255), FILLED);
            // Compute the bounding box
        Rect bbox(circles[i][0] - circles[i][2], circles[i][1] - circles[i][2], 2 * circles[i][2], 2 * circles[i][2]);
            // Create a black image
        Mat3b res(img.size(), Vec3b(0, 0, 0));
            // Copy only the image under the white circle to black image
        img.copyTo(res, mask);
            // Crop according to the roi
        res = res(bbox);
            // Save the image
        String name = "C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/hough_circles/c" + to_string(i) + ".png";
        imwrite(name, res);

        //watershed(res);

    }
    imshow("m1", img);

    waitKey(0);
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


    Mat featureMatrix(numPixels, 3, CV_32F);
    int index = 0;

    Mat hsv;
    cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV

    for (int i = 0; i < mean_shift_img.rows; i++) {
        for (int j = 0; j < mean_shift_img.cols; j++) {
            Vec3b pixel = mean_shift_img.at<Vec3b>(i, j);
            if (pixel != Vec3b(0, 0, 0)) {
                //featureMatrix.at<float>(index, 0) = static_cast<float>(pixel[0]);  // Blue channel
                //featureMatrix.at<float>(index, 1) = static_cast<float>(pixel[1]);  // Green channel
                //featureMatrix.at<float>(index, 2) = static_cast<float>(pixel[2]);  // Red channel
                featureMatrix.at<float>(index, 0) = static_cast<float>(i / mean_shift_img.rows);      // Pixel row
                featureMatrix.at<float>(index, 1) = static_cast<float>(j / mean_shift_img.cols);      // Pixel column
                featureMatrix.at<float>(index, 2) = static_cast<float>(hsv.at<Vec3b>(i, j)[0] / 180);     // H hsv color
                index++;
            }
        }
    }
    // Convert the reshaped image to floats
    Mat featureMatrixFloat;
    featureMatrix.convertTo(featureMatrixFloat, CV_32F);
    //compute kmeans
    Mat labels, centers;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);
    kmeans(featureMatrixFloat, numRegions, labels, criteria, 5, KMEANS_PP_CENTERS, centers);


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