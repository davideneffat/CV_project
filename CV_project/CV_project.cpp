
#include <opencv2\opencv.hpp>
#include <vector>
#include "Image_processing.h"

using namespace std;
using namespace cv;

void reconstruct_images(int N, int dataset_size) {

    vector<Mat> images; //vector of images

    for (int i = 1; i <= dataset_size; i++) {
        string path = "C:/Users/david/OneDrive/Desktop/Food_leftover_dataset/dataset/pasta_pesto/im" + to_string(i) + ".jpg";
        images.push_back(imread(path));
    }

    int patch_num = 0;  //patches counter
    for (int im = 0; im < images.size(); im++) {
        // Create a black image
        Mat res = Mat::zeros(Size(images[im].cols, images[im].rows), CV_8UC1);  //8 bit, 1 channel

        for (int r = 0; r < images[im].rows; r += N)
        {
            for (int c = 0; c < images[im].cols; c += N)
            {
                string path = "C:/Users/david/OneDrive/Desktop/Food_leftover_dataset/dataset/pasta_pesto_predicted128x128/im" + to_string(patch_num) + ".png";
                Mat patch = imread(path, IMREAD_GRAYSCALE);
                if ((r + patch.size().height < images[im].size().height) && (c + patch.size().width < images[im].size().width)) {
                    
                    patch.copyTo(res(Rect(c, r, patch.size().width, patch.size().height)));
                    patch_num++;
                }
            }
        }

        String output_path = "C:/Users/david/OneDrive/Desktop/Food_leftover_dataset/dataset/pasta_pesto_predicted/im" + to_string(im) + ".png";
        imwrite(output_path, res);
    }

}

void split_images(int N, int dataset_size) {
    vector<Mat> images; //vector of images

    for (int i = 1; i <= dataset_size; i++) {
        string path = "C:/Users/david/OneDrive/Desktop/Food_leftover_dataset/dataset/pasta_pesto_mask/m" + to_string(i) + ".png";
        images.push_back(imread(path));
    }

    int name = 0;
    for (int im = 0; im < images.size(); im++)
    {
        for (int r = 0; r < images[im].rows; r += N)
        {
            for (int c = 0; c < images[im].cols; c += N)
            {
                string path = "C:/Users/david/OneDrive/Desktop/Food_leftover_dataset/dataset/pasta_pesto_mask128x128/im" + to_string(name) + ".png";

                Mat patch = images[im](Range(r, min(r + N, images[im].rows)), Range(c, min(c + N, images[im].cols)));
                if ((patch.size().height == N) && (patch.size().width == N)) {
                    imwrite(path, patch);     //for saving new patches
                    name++;
                }
            }
        }
    }
}

vector<int> get_img_histogram(Mat img) {
    vector<int> histogram(256, 0);  //size=255, all elements set to 0
    int val = 0;
    for (int r = 0; r < img.rows; r += 1)
    {
        for (int c = 0; c < img.cols; c += 1)
        {
            val = img.at<uchar>(r, c);
            histogram[val] = histogram[val] + 1;
        }
    }
    return histogram;
}








int main(int argc, char* argv[])
{
    //split_images(128, 18);
    //reconstruct_images(128,18);

    //hough_transform("C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/imgs/im8.jpg");

    
    //bilateral con sigma s=2/4 e sigma h=10/20
    Mat src = imread("C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/hough_circles/h8.png");
    imshow("input src", src);
    

    //KMEANS PREPROCESSING:
    Mat bilateral_img;
    bilateralFilter(src, bilateral_img, 20, 200, 250, BORDER_DEFAULT);    //apply bilateral filter
    //imshow("Bilateral filter", bilateral_img);

    Mat mean_shift_img;
    pyrMeanShiftFiltering(bilateral_img, mean_shift_img, 20, 45, 3); //apply mean-shift
    imshow("Mean-shift", mean_shift_img);

    //Remove the dish (white pixels):
    Scalar lowerBound = Scalar(0, 0, 0); // H=colore[0,180], S=bianco-colore[0,255], V=nero-colore[0,255] to remove plate (bianco)
    Scalar upperBound = Scalar(255, 35, 255);
    Mat hsv;
    cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV
    Mat mask;
    inRange(hsv, lowerBound, upperBound, mask);    //filter pixels in range

    mean_shift_img.setTo(Scalar(0, 0, 0), mask);
    imshow("plate removal", mean_shift_img);


    erode(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);   //remove small isolated parts
    dilate(mean_shift_img, mean_shift_img, Mat(), Point(-1, -1), 2, 1, 1);
    imshow("erosion+dilation", mean_shift_img);




    //apply kmeans with k=1,2,3 and find best value for k
    Mat res = kmeans(mean_shift_img, 1);
    //print_clustered_img(res);
    int err1 = evaluate_kmeans(mean_shift_img, res, 1);
    cout << "Errore con 1 cluster = " << to_string(err1) << "\n";

    res = kmeans(mean_shift_img, 2);
    //print_clustered_img(res);
    int err2 = evaluate_kmeans(mean_shift_img, res, 2);
    cout << "Errore con 2 cluster = " << to_string(err2) << "\n";

    res = kmeans(mean_shift_img, 3);
    //print_clustered_img(res);
    int err3 = evaluate_kmeans(mean_shift_img, res, 3);
    cout << "Errore con 3 cluster = " << to_string(err3) << "\n";
    
    int best_k; //number of clusters that minimizes variance
    if (err1 <= err2) {
        if (err1 <= err3)
            best_k = 1;
        best_k = 3;
    }
    else if (err2 <= err3)
        best_k = 2;
    else
        best_k = 3;

    cout << to_string(best_k) << "\n";

    waitKey(0);
}