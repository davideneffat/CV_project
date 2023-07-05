
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

    vector<Mat> dishes = hough_transform("Food_leftover_dataset/start_imgs/imgs/im1.jpg");
    cout << "Finded " << to_string(dishes.size()) << " dishes\n";

    //Mat src = imread("C:/Users/david/source/repos/CV_project/CV_project/Food_leftover_dataset/start_imgs/hough_circles/h2.png");
    vector<vector<float>> colors;
    /*
    *colors = [im1_c1, im1_c2]
    *         [im2_c1]
    *         [im3_c1, im3_c2, im3_c3]
    * where im are the images coresponding to the dishes of the vassoio, c are the clusters finded in each dishes
    * and the values of the matrix are the mean colors (H of hsv value) of each cluster
    */

    for (int i = 0; i < dishes.size(); i++) {

        Mat src = dishes[i];
        string name = "input src" + to_string(i);
        imshow(name, src);

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
        err2 = err2 * 2;
        cout << "Errore con 2 cluster = " << to_string(err2) << "\n";

        Mat res3 = kmeans(mean_shift_img, 3);
        Mat out3 = print_clustered_img(res3);
        name = "k=3_" + to_string(i);
        imshow(name, out3);
        int err3 = evaluate_kmeans(mean_shift_img, res3, 3);
        err3 = err3 * 3;
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

    }

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

    

    waitKey(0);
}