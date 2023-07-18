
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




int main()
{
    



    //FAST E SIFT DETECTOR E DESCRIPTOR
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
    


    string path = "Food_leftover_dataset/start_imgs/imgs/im4.jpg";
    imshow("start img", imread(path));

    vector<Mat> dishes = hough_transform(path);
    cout << "Finded " << to_string(dishes.size()) << " dishes\n";

    vector<Mat> insalate = find_salad(path);
    cout << "Finded " << to_string(insalate.size()) << " insalate\n";
    for (int i = 0; i < insalate.size(); i++) {
        Mat src = insalate[i];
        string name = "insalata src" + to_string(i);
        imshow(name, src);
    }

    vector<vector<float>> foods;  //n=1 if food n present



    for (int i = 0; i < dishes.size(); i++) {

        Mat src = dishes[i];
        string name = "input src" + to_string(i);
        imshow(name, src);

        vector<float> foods_dish(14);

        Mat mean_shift_img = preprocess(src);   //image preprocessing
        name = "preprocessed" + to_string(i);
        imshow(name, mean_shift_img);

        Mat hsv;
        cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV


        Mat res;
        res = find_pasta(hsv);
        cout << "num pasta = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[0] = count_pixels(res, mean_shift_img);
        res = find_pesto(hsv);
        cout << "num pesto = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[1] = count_pixels(res, mean_shift_img);
        res = find_tomato(hsv);
        cout << "num pasta tomate = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[2] = count_pixels(res, mean_shift_img);
        res = find_meat_sauce(hsv);
        cout << "num pasta ragu = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[3] = count_pixels(res, mean_shift_img);
        res = find_pasta_clams_mussels(hsv);
        cout << "num pasta clams mussels = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[4] = count_pixels(res, mean_shift_img);
        res = find_rice(hsv);
        cout << "num rice = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[5] = count_pixels(res, mean_shift_img);
        res = find_grilled_pork_cutlet(hsv);
        cout << "num grilled pork cutlet = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[6] = count_pixels(res, mean_shift_img);
        res = find_fish_cutlet(hsv);
        cout << "num fish cutlet = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[7] = count_pixels(res, mean_shift_img);
        res = find_rabbit(hsv);
        cout << "num rabbit = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[8] = count_pixels(res, mean_shift_img);
        res = find_seafood_salad(hsv);
        cout << "num seafood salad = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[9] = count_pixels(res, mean_shift_img);
        res = find_beans(hsv);
        cout << "num beans = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[10] = count_pixels(res, mean_shift_img);
        res = find_potatoes(hsv);
        cout << "num potatoes = " << to_string(count_pixels(res, mean_shift_img)) << "\n";
        foods_dish[11] = count_pixels(res, mean_shift_img);



        foods.push_back(foods_dish);
        /*
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

        cout << to_string(findBestNumClusters(src)) << "\n";*/




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
    for (int i = 0; i < foods.size(); i++) {
        for (int j = 0; j < foods[i].size(); j++) {
            cout << to_string(foods[i][j]) << "  ";
        }
        cout << "\n";
    }

    for (int i = 0; i < foods.size(); i++) {
        if (foods[i][0] > 0.4) {
            foods[i][1] = foods[i][0] + foods[i][1];    //pasta + pesto
            foods[i][2] = foods[i][0] + foods[i][2];    //pasta + tomate
            foods[i][3] = foods[i][0] + foods[i][3];    //pasta + meat sauce
            foods[i][0] = 0.0;
        }
        else if (foods[i][0] < 0.4) {
            foods[i][1] = 0.0;    //pasta + pesto
            foods[i][2] = 0.0;    //pasta + tomate
            foods[i][3] = 0.0;    //pasta + meat sauce
            foods[i][0] = 0.0;
        }
        else if((foods[i][1] < 0.1) && (foods[i][2] < 0.1) && (foods[i][3] < 0.1))
            foods[i][0] = 0.0;
    }

    vector<float> food_highest_pixels;
    vector<int> food_highest_id;
    //find primo:
    for (int i = 0; i < foods.size(); i++) {
        float max = 0;
        int id = 0;
        for (int j = 0; j < 6; j++) {
            if (foods[i][j] > max) {
                max = foods[i][j];
                id = j;
            }
        }
        food_highest_pixels.push_back(max);
        food_highest_id.push_back(id);
    }

    for (int i = 0; i < food_highest_id.size(); i++) {
        cout << to_string(food_highest_id[i]) << " " << to_string(food_highest_pixels[i]) << "\n";
    }
    
    int first_dish_img = 0;

    float max = 0;
    int id = 0;
    for (int i = 0; i < food_highest_id.size(); i++) {
        if (food_highest_pixels[i] > max) {
            max = food_highest_pixels[i];
            id = food_highest_id[i];
            first_dish_img = i;
        }
    }
    cout << "First dish = " << to_string(id) << " with quantity = " << to_string(max) << "\n";


    int second_dish_img;
    if (first_dish_img == 0)
        second_dish_img = 1;    //there are at least 2 dishes
    else
        second_dish_img = 0;


    vector<int> food_pixels(14);
    Mat first = preprocess(dishes[first_dish_img]);
    food_pixels[id] = count_pixels_not_zero(first);    //update vector with number of pixels for first dish





    //find secondo e contorni:

    Mat second = preprocess(dishes[second_dish_img]);

    for (int i = 0; i < foods.size(); i++) {
        if (i == second_dish_img) {     //skip first dishes already founded
            if ((foods[i][10] > 0.15) && (foods[i][11] > 0.15)) { //beans and potatoes present
                cout << "Un secondo e due contorni trovati\n";
                Mat clustered = kmeans(second, 3);
                imshow("secondo", print_clustered_img(clustered));
                vector<float> mean_colors = calc_mean_cluster_color(second, clustered, 3);
                
                int patate;
                int fagioli;
                int altro;

                if ((mean_colors[1] > mean_colors[2]) && (mean_colors[1] > mean_colors[3]))  //patate con label 1
                    patate = 1;
                else if ((mean_colors[2] > mean_colors[1]) && (mean_colors[2] > mean_colors[3]))  //patate con label 2
                    patate = 2;
                else if ((mean_colors[3] > mean_colors[1]) && (mean_colors[3] > mean_colors[2]))  //patate con label 3
                    patate = 3;
                food_pixels[11] = count_pixels_with_value_n(clustered, patate);    //update vector with number of pixels for patate

                if ((mean_colors[1] < mean_colors[2]) && (mean_colors[1] < mean_colors[3]))  //fagioli con label 1
                    fagioli = 1;
                else if ((mean_colors[2] < mean_colors[1]) && (mean_colors[2] < mean_colors[3]))  //fagioli con label 2
                    fagioli = 2;
                else if ((mean_colors[3] < mean_colors[1]) && (mean_colors[3] < mean_colors[2]))  //fagioli con label 3
                    fagioli = 3;
                food_pixels[10] = count_pixels_with_value_n(clustered, fagioli);    //update vector with number of pixels for fagioli
                
                altro = 6 - patate - fagioli;

                
                float pork_pixels = foods[second_dish_img][6];
                float fish_pixels = foods[second_dish_img][7];
                float rabbit_pixels = foods[second_dish_img][8];
                float seafood_pixels = foods[second_dish_img][9];
                if ((pork_pixels > fish_pixels) && (pork_pixels > rabbit_pixels) && (pork_pixels > seafood_pixels))
                    food_pixels[6] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((fish_pixels > pork_pixels) && (fish_pixels > rabbit_pixels) && (fish_pixels > seafood_pixels))
                    food_pixels[7] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((rabbit_pixels > pork_pixels) && (rabbit_pixels > fish_pixels) && (rabbit_pixels > seafood_pixels))
                    food_pixels[8] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((seafood_pixels > pork_pixels) && (seafood_pixels > fish_pixels) && (seafood_pixels > rabbit_pixels))
                    food_pixels[9] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
            }

            else if (foods[i][10] > 0.3) {  //beans present
                cout << "Un secondo e fagioli di contorno trovati\n";
                Mat clustered = kmeans(second, 2);
                imshow("secondo", print_clustered_img(clustered));
                vector<float> mean_colors = calc_mean_cluster_color(second, clustered, 2);

                int fagioli;
                int altro;

                if (mean_colors[1] < mean_colors[2])
                    fagioli = 1;
                if (mean_colors[1] > mean_colors[2])
                    fagioli = 2;
                food_pixels[10] = count_pixels_with_value_n(clustered, fagioli);    //update vector with number of pixels for fagioli

                altro = 3 - fagioli;

                float pork_pixels = foods[second_dish_img][6];
                float fish_pixels = foods[second_dish_img][7];
                float rabbit_pixels = foods[second_dish_img][8];
                float seafood_pixels = foods[second_dish_img][9];
                if ((pork_pixels > fish_pixels) && (pork_pixels > rabbit_pixels) && (pork_pixels > seafood_pixels))
                    food_pixels[6] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((fish_pixels > pork_pixels) && (fish_pixels > rabbit_pixels) && (fish_pixels > seafood_pixels))
                    food_pixels[7] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((rabbit_pixels > pork_pixels) && (rabbit_pixels > fish_pixels) && (rabbit_pixels > seafood_pixels))
                    food_pixels[8] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((seafood_pixels > pork_pixels) && (seafood_pixels > fish_pixels) && (seafood_pixels > rabbit_pixels))
                    food_pixels[9] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
            }

            else if (foods[i][11] > 0.3) {  //potatoes present
                cout << "Un secondo e patate di contorno trovati\n";
                Mat clustered = kmeans(second, 2);
                imshow("secondo", print_clustered_img(clustered));
                vector<float> mean_colors = calc_mean_cluster_color(second, clustered, 2);

                int patate;
                int altro;

                if (mean_colors[1] > mean_colors[2])
                    patate = 1;
                if (mean_colors[1] < mean_colors[2])
                    patate = 2;
                food_pixels[11] = count_pixels_with_value_n(clustered, patate);    //update vector with number of pixels for fagioli

                altro = 3 - patate;

                float pork_pixels = foods[second_dish_img][6];
                float fish_pixels = foods[second_dish_img][7];
                float rabbit_pixels = foods[second_dish_img][8];
                float seafood_pixels = foods[second_dish_img][9];
                if ((pork_pixels > fish_pixels) && (pork_pixels > rabbit_pixels) && (pork_pixels > seafood_pixels))
                    food_pixels[6] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((fish_pixels > pork_pixels) && (fish_pixels > rabbit_pixels) && (fish_pixels > seafood_pixels))
                    food_pixels[7] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((rabbit_pixels > pork_pixels) && (rabbit_pixels > fish_pixels) && (rabbit_pixels > seafood_pixels))
                    food_pixels[8] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                if ((seafood_pixels > pork_pixels) && (seafood_pixels > fish_pixels) && (seafood_pixels > rabbit_pixels))
                    food_pixels[9] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
            }
            else {  //nessun contorno, solo secondo
                cout << "Un secondo e nessun contorno trovati\n";
                Mat clustered = kmeans(second, 1);
                imshow("secondo", print_clustered_img(clustered));

                float pork_pixels = foods[second_dish_img][6];
                float fish_pixels = foods[second_dish_img][7];
                float rabbit_pixels = foods[second_dish_img][8];
                float seafood_pixels = foods[second_dish_img][9];
                if((pork_pixels > fish_pixels) && (pork_pixels > rabbit_pixels) && (pork_pixels > seafood_pixels))
                    food_pixels[6] = count_cluster_pixels(clustered);    //update vector with number of pixels for second dish
                if ((fish_pixels > pork_pixels) && (fish_pixels > rabbit_pixels) && (fish_pixels > seafood_pixels))
                    food_pixels[7] = count_cluster_pixels(clustered);    //update vector with number of pixels for second dish
                if ((rabbit_pixels > pork_pixels) && (rabbit_pixels > fish_pixels) && (rabbit_pixels > seafood_pixels))
                    food_pixels[8] = count_cluster_pixels(clustered);    //update vector with number of pixels for second dish
                if ((seafood_pixels > pork_pixels) && (seafood_pixels > fish_pixels) && (seafood_pixels > rabbit_pixels))
                    food_pixels[9] = count_cluster_pixels(clustered);    //update vector with number of pixels for second dish
            }
        }
    }


    for (int i = 0; i < food_pixels.size(); i++) {
        cout << to_string(i)<< "= " << to_string(food_pixels[i]) << "  ";
    }
    cout << "\n";


    waitKey(0);
}