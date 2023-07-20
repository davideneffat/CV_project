
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "Image_processing.h"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>




using namespace std;
using namespace cv;



int main()
{
    

    string path = "Food_leftover_dataset/tray1/";  //image at begin of meal

    Mat source = imread(path+ "food_image.jpg");    //image at begin of meal
    Mat end_source = imread(path + "leftover2.jpg");    //image at end of meal

    Mat clusters(source.rows, source.cols, CV_8UC3, Scalar(0,0,0));
    Mat end_clusters(source.rows, source.cols, CV_8UC3, Scalar(0,0,0));

    namedWindow("start img", WINDOW_NORMAL);
    imshow("start img", source);

    vector<Mat> dishes = hough_transform(source);
    cout << "Finded " << to_string(dishes.size()) << " dishes\n";

    if (dishes.size() != 2) {
        cout << "Errore non è stato trovato almeno un primo e un secondo\n";
        waitKey(0);
        return 0;
    }

    vector<vector<float>> foods;  //n=1 if food n present

    vector<vector<int>> foods_localization;
    vector<vector<int>> end_foods_localization;

    for (int i = 0; i < dishes.size(); i++) {

        Mat src = dishes[i];
        string name = "start dish " + to_string(i);
        namedWindow(name, WINDOW_NORMAL);
        imshow(name, src);

        vector<float> foods_dish(14);

        Mat mean_shift_img = preprocess(src);   //image preprocessing
        name = "start preprocessed dish" + to_string(i);
        imshow(name, mean_shift_img);

        Mat hsv;
        cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV


        Mat res;
        res = find_pasta(hsv);
        foods_dish[0] = count_pixels(res, mean_shift_img);
        res = find_pesto(hsv);
        foods_dish[1] = count_pixels(res, mean_shift_img);
        res = find_tomato(hsv);
        foods_dish[2] = count_pixels(res, mean_shift_img);
        res = find_meat_sauce(hsv);
        foods_dish[3] = count_pixels(res, mean_shift_img);
        res = find_pasta_clams_mussels(hsv);
        foods_dish[4] = count_pixels(res, mean_shift_img);
        res = find_rice(hsv);
        foods_dish[5] = count_pixels(res, mean_shift_img);
        res = find_grilled_pork_cutlet(hsv);
        foods_dish[6] = count_pixels(res, mean_shift_img);
        res = find_fish_cutlet(hsv);
        foods_dish[7] = count_pixels(res, mean_shift_img);
        res = find_rabbit(hsv);
        foods_dish[8] = count_pixels(res, mean_shift_img);
        res = find_seafood_salad(hsv);
        foods_dish[9] = count_pixels(res, mean_shift_img);
        res = find_beans(hsv);
        foods_dish[10] = count_pixels(res, mean_shift_img);
        res = find_potatoes(hsv);
        foods_dish[11] = count_pixels(res, mean_shift_img);



        foods.push_back(foods_dish);


    }

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
    Mat cluster_primo = kmeans(first, 1);   //clusterize dish




    Mat template_img = dishes[first_dish_img];

    //let's cut template to remove black pixels around it
    int x_begin = 0;
    int x_end = first.cols;
    int y_begin = 0;
    int y_end = first.rows;
    for (int i = 0; i < first.rows; i++) {
        for (int j = 0; j < first.cols; j++) {
            if (first.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                y_end = i;
        }
    }
    for (int j = 0; j < first.cols; j++) {
        for (int i = 0; i < first.rows; i++) {
            if (first.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                x_end = j;
        }
    }
    for (int i = first.rows - 1; i >= 0; i--) {
        for (int j = first.cols - 1; j >= 0; j--) {
            if (first.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                y_begin = i;
        }
    }
    for (int j = first.cols - 1; j >= 0; j--) {
        for (int i = first.rows - 1; i >= 0; i--) {
            if (first.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
                x_begin = j;
        }
    }

    template_img = template_img(Range(y_begin, y_end), Range(x_begin, x_end));

    Point matchLoc = food_recognition_rectangle(source, template_img);
    rectangle(source, matchLoc, Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), Vec3b(255, 0, 0), 2, 8, 0);

    vector<int> primo_locations = { id, matchLoc.x, matchLoc.y, matchLoc.x + template_img.cols, matchLoc.y + template_img.rows };
    foods_localization.push_back(primo_locations);



    for (int i = 0; i < cluster_primo.rows; i++) {
        for (int j = 0; j < cluster_primo.cols; j++) {
            if (cluster_primo.at<Vec3b>(i, j)[0] == 1) {
                clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(id, id, id);
            }
        }
    }





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
            
                int secondo_id = 0;
                for (int id = 6; id < 10; id++) {
                    if (food_pixels[id] > 0)
                        secondo_id = id;
                }

                Point matchLoc = food_recognition_rectangle(source, dishes[second_dish_img]);

                for (int k = 1; k < 4; k++) {   //k = label of cluster
                    vector<Point> locations = calculate_template_from_cluster(source, clustered, dishes[second_dish_img], k);
                    rectangle(source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);
                    if (patate == k) {
                        vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                                }
                            }
                        }
                    }
                    else if (fagioli == k) {
                        vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                                }
                            }
                        }
                    }
                    else {
                        vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                                }
                            }
                        }
                    }
                }
            
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
            

                int secondo_id = 0;
                for (int id = 6; id < 10; id++) {
                    if (food_pixels[id] > 0)
                        secondo_id = id;
                }

                Point matchLoc = food_recognition_rectangle(source, dishes[second_dish_img]);

                for (int k = 1; k < 3; k++) {
                    vector<Point> locations = calculate_template_from_cluster(source, clustered, dishes[second_dish_img], k);
                    rectangle(source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

                    if (fagioli == k) {
                        vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                                }
                            }
                        }
                    }
                    else {
                        vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                                }
                            }
                        }
                    }
                }
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
            
                int secondo_id = 0;
                for (int id = 6; id < 10; id++) {
                    if (food_pixels[id] > 0)
                        secondo_id = id;
                }

                Point matchLoc = food_recognition_rectangle(source, dishes[second_dish_img]);

                for (int k = 1; k < 3; k++) {
                    vector<Point> locations = calculate_template_from_cluster(source, clustered, dishes[second_dish_img], k);
                    rectangle(source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

                    if (patate == k) {
                        vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                                }
                            }
                        }
                    }
                    else {
                        vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                        foods_localization.push_back(secondo_locations);

                        for (int i = 0; i < clustered.rows; i++) {
                            for (int j = 0; j < clustered.cols; j++) {
                                if (clustered.at<Vec3b>(i, j)[0] == k) {
                                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                                }
                            }
                        }
                    }
                }
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
            

                vector<Point> locations = calculate_template_from_cluster(source, clustered, dishes[second_dish_img], 1);
                rectangle(source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

                int secondo_id = 0;
                for (int id = 6; id < 10; id++) {
                    if (food_pixels[id] > 0)
                        secondo_id = id;
                }
                vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                foods_localization.push_back(secondo_locations);

                Point matchLoc = food_recognition_rectangle(source, dishes[second_dish_img]);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == 1) {
                            clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                        }
                    }
                }
            }
        }
    }


    //find insalata:
    vector<Mat> insalate = find_salad(source);
    cout << "Finded " << to_string(insalate.size()) << " insalate\n";
    for (int i = 0; i < insalate.size(); i++) {
        Mat src = insalate[i];
        string name = "start insalata";
        imshow(name, src);
        food_pixels[12] = count_pixels_not_zero(src);

        Mat template_img = src;
        Point matchLoc = food_recognition_rectangle(source, template_img);
        rectangle(source, matchLoc, Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), Vec3b(0, 255, 0), 2, 8, 0);

        vector<int> insalata_locations = { 12, matchLoc.x, matchLoc.y, matchLoc.x + template_img.cols, matchLoc.y + template_img.rows };
        foods_localization.push_back(insalata_locations);

        for (int i = 0; i < template_img.rows; i++) {
            for (int j = 0; j < template_img.cols; j++) {
                if (template_img.at<Vec3b>(i, j)[0] != 0) {
                    clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(12, 12, 12);
                }
            }
        }
    }




    namedWindow("localized img", WINDOW_NORMAL);
    imshow("localized img", source);


    namedWindow("clustered img", WINDOW_NORMAL);
    imshow("clustered img", print_clustered_img(clusters));



    for (int i = 0; i < food_pixels.size(); i++) {
        cout << to_string(i)<< "= " << to_string(food_pixels[i]) << "  ";
    }
    cout << "\n";



    cout << "Bounding box at begin of meal:\n";
    for (int i = 0; i < foods_localization.size(); i++) {
        cout << "ID: " << to_string(foods_localization[i][0]) << "; [ ";
        for (int j = 1; j < foods_localization[i].size(); j++) {
            cout << to_string(foods_localization[i][j]) << " ";
        }
        cout << "]\n";
    }




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //WORK ON TRAY AT END OF MEAL:



    namedWindow("end img", WINDOW_NORMAL);
    imshow("end img", end_source);

    vector<Mat> end_dishes = hough_transform(end_source); //vector of dishes extracted with hough transform
    cout << "Finded " << to_string(end_dishes.size()) << " dishes\n";

    if (end_dishes.size() != 2) {
        cout << "Errore non è stato trovato almeno un primo e un secondo nei piatti alla fine del pasto\n";
        waitKey(0);
        return 0;
    }

    vector<vector<float>> end_foods;  //n=1 if food n present    (vector of vector)

    vector<int> food_pixels_remained(14);



    for (int i = 0; i < end_dishes.size(); i++) {

        Mat src = end_dishes[i];
        string name = "end dish " + to_string(i);
        namedWindow(name, WINDOW_NORMAL);
        imshow(name, src);

        vector<float> foods_dish(14);

        Mat mean_shift_img = preprocess(src);   //image preprocessing
        name = "end preprocessed dish" + to_string(i);
        imshow(name, mean_shift_img);

        Mat hsv;
        cvtColor(mean_shift_img, hsv, COLOR_BGR2HSV);  //transform from RGB to HSV


        Mat res;
        res = find_pasta(hsv);
        foods_dish[0] = count_pixels(res, mean_shift_img);
        food_pixels_remained[0] = count_cluster_pixels(res);
        res = find_pesto(hsv);
        foods_dish[1] = count_pixels(res, mean_shift_img);
        food_pixels_remained[1] = count_cluster_pixels(res);
        res = find_tomato(hsv);
        foods_dish[2] = count_pixels(res, mean_shift_img);
        food_pixels_remained[2] = count_cluster_pixels(res);
        res = find_meat_sauce(hsv);
        foods_dish[3] = count_pixels(res, mean_shift_img);
        food_pixels_remained[3] = count_cluster_pixels(res);
        res = find_pasta_clams_mussels(hsv);
        foods_dish[4] = count_pixels(res, mean_shift_img);
        food_pixels_remained[4] = count_cluster_pixels(res);
        res = find_rice(hsv);
        foods_dish[5] = count_pixels(res, mean_shift_img);
        food_pixels_remained[5] = count_cluster_pixels(res);
        res = find_grilled_pork_cutlet(hsv);
        foods_dish[6] = count_pixels(res, mean_shift_img);
        food_pixels_remained[6] = count_cluster_pixels(res);
        res = find_fish_cutlet(hsv);
        foods_dish[7] = count_pixels(res, mean_shift_img);
        food_pixels_remained[7] = count_cluster_pixels(res);
        res = find_rabbit(hsv);
        foods_dish[8] = count_pixels(res, mean_shift_img);
        food_pixels_remained[8] = count_cluster_pixels(res);
        res = find_seafood_salad(hsv);
        foods_dish[9] = count_pixels(res, mean_shift_img);
        food_pixels_remained[9] = count_cluster_pixels(res);
        res = find_beans(hsv);
        foods_dish[10] = count_pixels(res, mean_shift_img);
        food_pixels_remained[10] = count_cluster_pixels(res);
        res = find_potatoes(hsv);
        foods_dish[11] = count_pixels(res, mean_shift_img);
        food_pixels_remained[11] = count_cluster_pixels(res);



        end_foods.push_back(foods_dish);


    }
    for (int i = 0; i < end_foods.size(); i++) {
        for (int j = 0; j < end_foods[i].size(); j++) {
            cout << to_string(end_foods[i][j]) << "  ";
        }
        cout << "\n";
    }

    vector<int> end_food_pixels(14);


    //find first dish at end of meal
    max = 0.0;
    int end_first_dish_img = 0;
    for (int i = 0; i < 6; i++) {
        if (food_pixels[i] > 0) {
            for (int j = 0; j < end_foods.size(); j++) {
                if (end_foods[j][i] > max) {
                    max = end_foods[j][i];
                    end_first_dish_img = j;     //0 or 1
                }
            }
            break;
        }
    }




    int end_second_dish_img;
    if (end_first_dish_img == 0)
        end_second_dish_img = 1;    //there are at least 2 dishes
    else
        end_second_dish_img = 0;

    Mat end_first = preprocess(end_dishes[end_first_dish_img]);
    GaussianBlur(end_first, end_first, Size(7,7), 3,0); //per ridurre l'effetto del sugo rimasto sul piatto

    int primo_id = 0;
    for (int i = 0; i < 6; i++) {
        if (food_pixels[i] > 0) {
            end_food_pixels[i] = count_pixels_not_zero(end_first);    //update vector with number of pixels for first dish
            primo_id = i;
        }
    }
    cluster_primo = kmeans(end_first, 1);


    template_img = end_dishes[end_first_dish_img];
    matchLoc = food_recognition_rectangle(end_source, template_img);
    rectangle(end_source, matchLoc, Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), Vec3b(255, 0, 0), 2, 8, 0);

    primo_locations = { primo_id, matchLoc.x, matchLoc.y, matchLoc.x + template_img.cols, matchLoc.y + template_img.rows };
    end_foods_localization.push_back(primo_locations);

    for (int i = 0; i < cluster_primo.rows; i++) {
        for (int j = 0; j < cluster_primo.cols; j++) {
            if (cluster_primo.at<Vec3b>(i, j)[0] == 1) {
                end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(id, id, id);
            }
        }
    }


    bool end_secondo = false;   //true if second is not completely eaten






    //find second dish at end of meal
    int k = 0;  //num of foods in second dish
    for (int i = 6; i < food_pixels.size(); i++) {
        if (food_pixels[i] > 0) {
            k++;
            //end_food_pixels[i] = food_pixels_remained[i];//end_foods[end_second_dish_img][i];
            if (end_foods[end_second_dish_img][i] > 0.05)   //second not completely eaten
                end_secondo = true;
        }//sono int, mentre gli altri float quindi approssima tutto a zero
    }

    bool end_patate = false;
    bool end_fagioli = false;
    if (end_foods[end_second_dish_img][10] > 0.05)  //fagioli pixels founded in end image
        if(food_pixels[10] > 0)     //fagioli pixels present in original image
            end_fagioli = true;
    if (end_foods[end_second_dish_img][11] > 0.05)
        if (food_pixels[11] > 0)
            end_patate = true;


    Mat end_second = preprocess(end_dishes[end_second_dish_img]);


    if (end_secondo && end_fagioli && end_patate) { //3 foods still remaining
        cout << "Foods remained in secon dish are beans, potatoes and the second\n";
        Mat clustered = kmeans(end_second, 3);
        imshow("end secondo", print_clustered_img(clustered));
        vector<float> mean_colors = calc_mean_cluster_color(end_second, clustered, 3);

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

        int secondo_id = 0;
        for (int i = 6; i < food_pixels.size(); i++) {
            if (food_pixels[i] > 0) {
                end_food_pixels[i] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                secondo_id = i;
                break;
            }
        }

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int k = 1; k < 4; k++) {
            vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], k);
            rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);
            if (patate == k) {
                vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                        }
                    }
                }
            }
            else if (fagioli == k) {
                vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                        }
                    }
                }
            }
            else {
                vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                        }
                    }
                }
            }
        }

    }
    else if (end_secondo && end_fagioli) {
        cout << "Foods remained in secon dish are beans and the second\n";
        Mat clustered = kmeans(end_second, 2);
        imshow("end secondo", print_clustered_img(clustered));
        vector<float> mean_colors = calc_mean_cluster_color(end_second, clustered, 2);

        int fagioli;
        int altro;

        if (mean_colors[1] < mean_colors[2])
            fagioli = 1;
        if (mean_colors[1] > mean_colors[2])
            fagioli = 2;
        end_food_pixels[10] = count_pixels_with_value_n(clustered, fagioli);    //update vector with number of pixels for fagioli

        altro = 3 - fagioli;

        int secondo_id = 0;
        for (int i = 6; i < food_pixels.size(); i++) {
            if (food_pixels[i] > 0) {
                end_food_pixels[i] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                secondo_id = i;
                break;
            }
        }

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int k = 1; k < 3; k++) {
            vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], k);
            rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

            if (fagioli == k) {
                vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                        }
                    }
                }
            }
            else {
                vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                        }
                    }
                }
            }
        }
    }
    else if (end_secondo && end_patate) {
        cout << "Foods remained in secon dish are potatoes and the second\n";
        Mat clustered = kmeans(end_second, 2);
        imshow("end secondo", print_clustered_img(clustered));
        vector<float> mean_colors = calc_mean_cluster_color(end_second, clustered, 2);

        int patate;
        int altro;

        if (mean_colors[1] > mean_colors[2])
            patate = 1;
        if (mean_colors[1] < mean_colors[2])
            patate = 2;
        end_food_pixels[11] = count_pixels_with_value_n(clustered, patate);    //update vector with number of pixels for patate

        altro = 3 - patate;

        int secondo_id = 0;
        for (int i = 6; i < food_pixels.size(); i++) {
            if (food_pixels[i] > 0) {
                end_food_pixels[i] = count_pixels_with_value_n(clustered, altro);    //update vector with number of pixels for second dish
                secondo_id = i;
                break;
            }
        }

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int k = 1; k < 3; k++) {
            vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], k);
            rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

            if (patate == k) {
                vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                        }
                    }
                }
            }
            else {
                vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                        }
                    }
                }
            }
        }
    }
    else if (end_fagioli && end_patate) {
        cout << "Foods remained in secon dish are beans and potatoes\n";
        Mat clustered = kmeans(end_second, 2);
        imshow("end secondo", print_clustered_img(clustered));
        vector<float> mean_colors = calc_mean_cluster_color(end_second, clustered, 2);

        int patate;
        int fagioli;

        if (mean_colors[1] > mean_colors[2])
            patate = 1;
        if (mean_colors[1] < mean_colors[2])
            patate = 2;
        end_food_pixels[11] = count_pixels_with_value_n(clustered, patate);    //update vector with number of pixels for patate

        fagioli = 3 - patate;

        end_food_pixels[10] = count_pixels_with_value_n(clustered, fagioli);    //update vector with number of pixels for fagioli

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int k = 1; k < 3; k++) {
            vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], k);
            rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

            if (patate == k) {
                vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                        }
                    }
                }
            }
            else {  //fagioli
                vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
                end_foods_localization.push_back(secondo_locations);

                for (int i = 0; i < clustered.rows; i++) {
                    for (int j = 0; j < clustered.cols; j++) {
                        if (clustered.at<Vec3b>(i, j)[0] == k) {
                            end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                        }
                    }
                }
            }
        }
    }
    else if (end_secondo) {
        cout << "Food remained in secon dish is the second\n";
        Mat clustered = kmeans(second, 1);
        imshow("secondo", print_clustered_img(clustered));

        for (int i = 6; i < food_pixels.size(); i++) {
            if (food_pixels[i] > 0) {
                end_food_pixels[i] = count_pixels_with_value_n(clustered, 1);    //update vector with number of pixels for second dish
                break;
            }
        }

        vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], 1);
        rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

        int secondo_id = 0;
        for (int id = 6; id < 10; id++) {
            if (food_pixels[id] > 0)
                secondo_id = id;
        }
        vector<int> secondo_locations = { secondo_id, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
        end_foods_localization.push_back(secondo_locations);

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int i = 0; i < clustered.rows; i++) {
            for (int j = 0; j < clustered.cols; j++) {
                if (clustered.at<Vec3b>(i, j)[0] == 1) {
                    end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(secondo_id, secondo_id, secondo_id);
                }
            }
        }
    }
    else if (end_fagioli) {
        cout << "Food remained in secon dish is beans\n";
        Mat clustered = kmeans(second, 1);
        imshow("secondo", print_clustered_img(clustered));

        end_food_pixels[10] = count_pixels_with_value_n(clustered, 1);    //update vector with number of pixels for fagioli


        vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], 1);
        rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

        vector<int> secondo_locations = { 10, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
        end_foods_localization.push_back(secondo_locations);

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int i = 0; i < clustered.rows; i++) {
            for (int j = 0; j < clustered.cols; j++) {
                if (clustered.at<Vec3b>(i, j)[0] == 1) {
                    end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(10, 10, 10);
                }
            }
        }

    }
    else if (end_patate) {
        cout << "Food remained in secon dish is potatoes\n";
        Mat clustered = kmeans(second, 1);
        imshow("secondo", print_clustered_img(clustered));

        end_food_pixels[11] = count_pixels_with_value_n(clustered, 1);    //update vector with number of pixels for patate


        vector<Point> locations = calculate_template_from_cluster(end_source, clustered, end_dishes[end_second_dish_img], 1);
        rectangle(end_source, locations[0], locations[1], Vec3b(0, 0, 255), 2, 8, 0);

        vector<int> secondo_locations = { 11, locations[0].x, locations[0].y, locations[1].x, locations[1].y };
        end_foods_localization.push_back(secondo_locations);

        Point matchLoc = food_recognition_rectangle(end_source, end_dishes[end_second_dish_img]);

        for (int i = 0; i < clustered.rows; i++) {
            for (int j = 0; j < clustered.cols; j++) {
                if (clustered.at<Vec3b>(i, j)[0] == 1) {
                    end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(11, 11, 11);
                }
            }
        }
    }



    if (food_pixels[12] > 0) {  //salad present at begin of meal
        //find insalata:
        vector<Mat> end_insalate = find_salad(end_source);
        cout << "Finded " << to_string(insalate.size()) << " insalate\n";
        if (insalate.size() != 1) {
            cout << "Insalata non trovata alla fine del pasto\n";
        }
        else {
            Mat src = end_insalate[0];
            string name = "end insalata";
            imshow(name, src);
            int pixels = find_remaining_salad(src);
            cout << "Finded " << to_string(pixels) << " remaining pixels of salad\n";
            end_food_pixels[12] = pixels;

            Mat template_img = src;
            Point matchLoc = food_recognition_rectangle(end_source, template_img);
            rectangle(end_source, matchLoc, Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows), Vec3b(0, 255, 0), 2, 8, 0);

            vector<int> insalata_locations = { 12, matchLoc.x, matchLoc.y, matchLoc.x + template_img.cols, matchLoc.y + template_img.rows };
            end_foods_localization.push_back(insalata_locations);

            for (int i = 0; i < template_img.rows; i++) {
                for (int j = 0; j < template_img.cols; j++) {
                    if (template_img.at<Vec3b>(i, j)[0] != 0) {
                        end_clusters.at<Vec3b>(matchLoc.y + i, matchLoc.x + j) = Vec3b(12, 12, 12);
                    }
                }
            }
        }
    }


    namedWindow("end localized img", WINDOW_NORMAL);
    imshow("end localized img", end_source);


    namedWindow("end clustered img", WINDOW_NORMAL);
    imshow("end clustered img", print_clustered_img(end_clusters));


    for (int i = 0; i < end_food_pixels.size(); i++) {
        cout << to_string(i) << "= " << to_string(end_food_pixels[i]) << "  ";
    }
    cout << "\n";


    cout << "Bounding box at end of meal:\n";
    for (int i = 0; i < end_foods_localization.size(); i++) {
        cout << "ID: " << to_string(foods_localization[i][0]) << "; [ ";
        for (int j = 1; j < end_foods_localization[i].size(); j++) {
            cout << to_string(end_foods_localization[i][j]) << " ";
        }
        cout << "]\n";
    }





    //EVALUATE METRICS

    Mat mask = imread(path + "masks/food_image_mask.png");

    int num_foods = 0;
    float sum_IoU = 0;
    for (int food = 0; food < food_pixels.size(); food++) {//vector with num of pixels for foods in image
        int intersection = 0;
        if (food_pixels[food] > 0) {    //food present
            num_foods++;
            for (int i = 0; i < clusters.rows; i++) {
                for (int j = 0; j < clusters.cols; j++) {
                    int mask_pixel = mask.at<Vec3b>(i, j)[0];
                    int predicted_pixel = clusters.at<Vec3b>(i, j)[0];
                    if ((mask_pixel == food)&&(predicted_pixel == food)) {
                        intersection++;
                    }
                }
            }
            int unione = count_pixels_with_value_n(clusters, food) + count_pixels_with_value_n(mask, food);
            float IoU = 0.0;
            IoU = (float)intersection / (float)(unione-intersection);
            cout << "IoU for class " << to_string(food) << " = " << to_string(IoU) << "\n";
            sum_IoU += IoU;
        }
    }
    float average_IoU = (float)sum_IoU / (float)num_foods;
    cout << "average IoU = " << to_string(average_IoU) << "\n";



    Mat end_mask = imread(path + "masks/leftover2.png");
    sum_IoU = 0;
    for (int food = 0; food < food_pixels.size(); food++) {//vector with num of pixels for foods in image
        int intersection = 0;
        if (food_pixels[food] > 0) {    //food present
            for (int i = 0; i < end_clusters.rows; i++) {
                for (int j = 0; j < end_clusters.cols; j++) {
                    int mask_pixel = end_mask.at<Vec3b>(i, j)[0];
                    int predicted_pixel = end_clusters.at<Vec3b>(i, j)[0];
                    if ((mask_pixel == food) && (predicted_pixel == food)) {
                        intersection++;
                    }
                }
            }
            int unione = count_pixels_with_value_n(end_clusters, food) + count_pixels_with_value_n(end_mask, food);
            float IoU = 0.0;
            IoU = (float)intersection / (float)(unione - intersection);
            cout << "end IoU for class " << to_string(food) << " = " << to_string(IoU) << "\n";
            sum_IoU += IoU;
        }
    }
    average_IoU = (float)sum_IoU / (float)num_foods;
    cout << "average IoU at end of meal = " << to_string(average_IoU) << "\n";


    //third metric:
    for (int food = 0; food < food_pixels.size(); food++) {//vector with num of pixels for foods in image
        int intersection = 0;
        if (food_pixels[food] > 0) {    //food present
            float ratio_r = (float)end_food_pixels[food] / (float)food_pixels[food];
            cout << "ratio R for class " << to_string(food) << " = " << to_string(ratio_r) << "\n";
        }
    }

    for (int food = 0; food < food_pixels.size(); food++) {//vector with num of pixels for foods in image
        int intersection = 0;
        if (food_pixels[food] > 0) {    //food present
            int start_pixels = count_pixels_with_value_n(mask, food);
            int end_pixels = count_pixels_with_value_n(end_mask, food);
            float ratio_r = (float)end_pixels / (float)start_pixels;
            cout << "real ratio R for class " << to_string(food) << " = " << to_string(ratio_r) << "\n";
        }
    }






    waitKey(0);
    return 0;
}