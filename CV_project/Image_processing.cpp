#include <opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

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

int find_remaining_salad(Mat image) {

    Scalar lowerBound = Scalar(0, 20, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255]
    Scalar upperBound = Scalar(45, 255, 255);

    Mat output_mask;
    inRange(image, lowerBound, upperBound, output_mask);    //filter pixels in range

    imshow("end salad", output_mask);
    //count pixel of output mask
    int res_pixels_black = 0;

    for (int i = 0; i < output_mask.rows; i++) {
        for (int j = 0; j < output_mask.cols; j++) {
            if (output_mask.at<uchar>(i, j) == 0)
                res_pixels_black += 1;
        }
    }
    int res_pixels_white = (output_mask.rows * output_mask.cols) - res_pixels_black;
    return  res_pixels_white;
}



vector<Mat> find_salad(Mat img) {

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
    HoughCircles(img_blur, circles, HOUGH_GRADIENT, 1, img2.rows / 2, 100, 45, 150, 230);

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
        

    }
    return dishes;
}

vector<Mat> hough_transform(Mat img) {

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

    //imshow("Image with cont",img2);
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


Point food_recognition_rectangle(Mat source, Mat template_img) {
    
    Mat img_display;
    Mat result;
    source.copyTo(img_display);
    int result_cols = source.cols - template_img.cols + 1;
    int result_rows = source.rows - template_img.rows + 1;
    result.create(result_rows, result_cols, CV_32F);
    matchTemplate(source, template_img, result, TM_SQDIFF);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    matchLoc = minLoc;
    //rectangle(img_display, matchLoc, Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), color, 2, 8, 0);
    return matchLoc;
}


vector<Point> calculate_template_from_cluster(Mat source, Mat clustered, Mat dish, int label) {
        Mat template_img(clustered.rows, clustered.cols, dish.type(), Scalar(0, 0, 0));
        for (int i = 0; i < clustered.rows; i++) {
            for (int j = 0; j < clustered.cols; j++) {
                if (clustered.at<Vec3b>(i, j)[0] == label) {
                    template_img.at<Vec3b>(i, j)[0] = dish.at<Vec3b>(i, j)[0];
                    template_img.at<Vec3b>(i, j)[1] = dish.at<Vec3b>(i, j)[1];
                    template_img.at<Vec3b>(i, j)[2] = dish.at<Vec3b>(i, j)[2];
                }
            }
        }
        //let's cut template to remove black pixels around it
        int x_begin = 0;
        int x_end = template_img.cols;
        int y_begin = 0;
        int y_end = template_img.rows;
        for (int i = 0; i < template_img.rows; i++) {
            for (int j = 0; j < template_img.cols; j++) {
                if (template_img.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                    y_end = i;
            }
        }
        for (int j = 0; j < template_img.cols; j++) {
            for (int i = 0; i < template_img.rows; i++) {
                if (template_img.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                    x_end = j;
            }
        }
        for (int i = template_img.rows - 1; i >= 0; i--) {
            for (int j = template_img.cols - 1; j >= 0; j--) {
                if (template_img.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                    y_begin = i;
            }
        }
        for (int j = template_img.cols - 1; j >= 0; j--) {
            for (int i = template_img.rows - 1; i >= 0; i--) {
                if (template_img.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                    x_begin = j;
            }
        }

        template_img = dish(Range(y_begin, y_end), Range(x_begin, x_end));

        //imshow("template" + to_string(label), template_img);
        Point matchLoc = food_recognition_rectangle(source, template_img);
        vector<Point> locations_output;
        locations_output.push_back(matchLoc);
        locations_output.push_back(Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows));
    return locations_output;
}


Mat preprocess(Mat img) {
    //KMEANS PREPROCESSING:
    Mat bilateral_img;
    bilateralFilter(img, bilateral_img, 20, 200, 250, BORDER_DEFAULT);    //apply bilateral filter

    Mat mean_shift_img;
    pyrMeanShiftFiltering(bilateral_img, mean_shift_img, 32, 16, 3); //apply mean-shift

    //Remove the plate (white pixels):
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
    /*cout << "Center matrix:\n";
    for (int i = 0; i < centers.rows; i++) {
        for (int j = 0; j < centers.cols; j++) {
            cout << to_string(centers.at<float>(i, j)) << " ";
        }
        cout << "\n";
    }*/



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







Mat print_clustered_img(Mat img) {

    Mat outputImage = Mat::zeros(img.size(), CV_8UC3);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            outputImage.at<Vec3b>(i, j)[0] = (img.at<Vec3b>(i, j)[0]) * 20;
            outputImage.at<Vec3b>(i, j)[1] = (img.at<Vec3b>(i, j)[1]) * 20;
            outputImage.at<Vec3b>(i, j)[2] = (img.at<Vec3b>(i, j)[2]) * 20;
        }
    }
    return outputImage;
}



