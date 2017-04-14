//Binary classification of images using SVN algorithm
//This code was written for university Computer Grafics course
//The task was to write function that extract features from image (HOG function)

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

const double pi = acos(-1);
const float eps = 0.000001;
// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);

    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Exatract features from dataset.
inline void HOG(BMP *image, vector <float> & one_image_features){

        uint h = image->TellHeight();
        uint w = image->TellWidth();
        RGBApixel pixel;

        //counting brightness
        vector <vector <float> > brigthtness(w, vector <float> (h));
        for(uint i = 0; i < w; i++){
            for(uint j = 0; j < h; j++){
                pixel = image->GetPixel(i, j);
                brigthtness[i][j] = 0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue;
            }
        }

        vector <vector <float> > g_sobelf(w, vector <float> (h)), v_sobelf(w, vector <float> (h));

        //sobel filter (g_sobel - horisontal, v_sobel - vertical)
        for(uint i = 0; i < w; i++){
            for(uint j = 0; j < h; j++){
                if(i > 0 && i < w - 1 && j > 0 && j < h - 1){
                    g_sobelf[i][j] = -1 * brigthtness[i - 1][j] + brigthtness[i + 1][j];
                    v_sobelf[i][j] = -1 * brigthtness[i][j - 1] + brigthtness[i][j + 1];
                }
                else{
                    g_sobelf[i][j] = 0;
                    v_sobelf[i][j] = 0;
                }
            }
        }

        //counting the absolute value and direction of gradient
        vector <vector <float> > abs_value(w, vector <float> (h)), direction(w, vector <float> (h));
        for(uint i = 0; i < w; i++){
            for(uint j = 0; j < h; j++){
                abs_value[i][j] = sqrt(g_sobelf[i][j] * g_sobelf[i][j] + v_sobelf[i][j] * v_sobelf[i][j]);
                if(fabs(g_sobelf[i][j] - 0.0) < eps){
                    if(v_sobelf[i][j] >= 0.0) direction[i][j] = pi / 2;
                    else direction[i][j] = -1 * pi / 2;
                }
                else{
                    if(g_sobelf[i][j] > 0.0) direction[i][j] = atan(v_sobelf[i][j] / g_sobelf[i][j]);
                    if(g_sobelf[i][j] < 0.0){
                        if(v_sobelf[i][j] >= 0.0) direction[i][j] = atan(v_sobelf[i][j] / g_sobelf[i][j]) + pi;
                        if(v_sobelf[i][j] < 0.0) direction[i][j] = atan(v_sobelf[i][j] / g_sobelf[i][j]) - pi;
                    }
                }
            }
        }

        int g_size, v_size, g_size_r, v_size_r; //sizes of segments (we divide image on 16*8 segments)
        g_size = w / 16;
        g_size_r = w / 16 + w % 16;
        v_size = h / 8;
        v_size_r = h / 8 + h % 8;

        vector <vector <float> > segments(128, vector <float> (16)); //segment from -pi to pi we divide into 16 parts

        for(int i = 0; i < 128; i++){
            for(int j = 0; j < 16; j++){
                segments[i][j] = 0.0;
            }
        }

        //start counting the histogramm
        for(int i = 0; i < 15; i++){
            for(int j = 0; j < 7; j++){

                for(int y = j * v_size; y < (j + 1) * v_size; y++){
                    for(int x = i * g_size; x < (i + 1) * g_size; x++){
                        segments[i * 8 + j][static_cast <int> (((direction[x][y] + pi) * 8) / pi)] += abs_value[x][y];
                    }
                }

            }
        }

        for(int j = 0; j < 7; j++){
            for(int y = j * v_size; y < (j + 1) * v_size; y++){
                for(int x = 15 * g_size; x < 15 * g_size + g_size_r; x++){
                    segments[120 + j][static_cast <int> (((direction[x][y] + pi) * 8) / pi)] += abs_value[x][y];
                }
            }
        }


        for(int i = 0; i < 15; i++){
            for(int y = 7 * v_size; y < 7 * v_size + v_size_r; y++){
                for(int x = i * g_size; x < (i + 1) * g_size; x++){
                    segments[i * 8 + 7][static_cast <int> (((direction[x][y] + pi) * 8) / pi)] += abs_value[x][y];
                }
            }
        }

        for(int y = 7 * v_size; y < 7 * v_size + v_size_r; y++){
            for(int x = 15 * g_size; x < 15 * g_size + g_size_r; x++){
                segments[127][static_cast <int> (((direction[x][y] + pi) * 8) / pi)] += abs_value[x][y];
            }
        }
        //the end of counting the histogramm

        //normalization
        float norm;
        for(int i = 0; i < 128; i++){
            norm = 0.0;
            for(int j = 0; j < 16; j++) norm += segments[i][j]*segments[i][j];
            norm = sqrt(norm);
            if(norm - 0.0 < eps) continue;
            for(int j = 0; j < 16; j++) segments[i][j] /= norm;
        }

        for(int i = 0; i < 16; i++)
            for(int j = 0; j < 8; j++)
                for(int c = 0; c < 16; c++) one_image_features.push_back(segments[i * 8 + j][c]);
}

void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        vector<float> one_image_features;

        //matrixes of average colours for colour features
        vector <vector <float>> red_av(8, vector <float> (8));
        vector <vector <float>> green_av(8, vector <float> (8));
        vector <vector <float>> blue_av(8, vector <float> (8));
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                red_av[i][j] = 0;
                green_av[i][j] = 0;
                blue_av[i][j] = 0;
            }
        }
        uint h = data_set[image_idx].first->TellHeight();
        uint w = data_set[image_idx].first->TellWidth();
        RGBApixel pixel;

        //start counting average colours
        int g_size = w/8, v_size = h/8, g_size_r = w/8 + w%8, v_size_r = h/8 + h%8;
        int square1 = g_size * v_size, square2 = g_size * v_size_r, square3 = g_size_r * v_size, square4 = g_size_r * v_size_r;
        for(int i = 0; i < 7; i++){
            for(int j = 0; j < 7; j++){
                for(int x = i * g_size; x < (i + 1) * g_size; x++){
                    for(int y = j * v_size; y < (j + 1)*v_size; y++){
                        pixel = data_set[image_idx].first->GetPixel(x, y);
                        red_av[i][j] += pixel.Red;
                        green_av[i][j] += pixel.Green;
                        blue_av[i][j] += pixel.Blue;
                    }
                }
                red_av[i][j] /= square1;
                green_av[i][j] /= square1;
                blue_av[i][j] /= square1;
            }
        }

        for(int i = 0; i < 7; i++){
            for(int x = i * g_size; x < (i + 1) * g_size; x++){
                for(int y = 7 * v_size; y < 7 * v_size + v_size_r; y++){
                    pixel = data_set[image_idx].first->GetPixel(x, y);
                    red_av[i][7] += pixel.Red;
                    green_av[i][7] += pixel.Green;
                    blue_av[i][7] += pixel.Blue;
                }
            }
            red_av[i][7] /= square2;
            green_av[i][7] /= square2;
            blue_av[i][7] /= square2;
        }

        for(int j = 0; j < 7; j++){
            for(int x = 7 * g_size; x < 7 * g_size + g_size_r; x++){
                for(int y = j * v_size; y < (j + 1) * v_size; y++){
                    pixel = data_set[image_idx].first->GetPixel(x, y);
                    red_av[7][j] += pixel.Red;
                    green_av[7][j] += pixel.Green;
                    blue_av[7][j] += pixel.Blue;
                }
            }
            red_av[7][j] /= square3;
            green_av[7][j] /= square3;
            blue_av[7][j] /= square3;
        }

        for(int x = 7 * g_size; x < 7 * g_size + g_size_r; x++){
            for(int y = 7 * v_size; y < 7 * v_size + v_size_r; y++){
                pixel = data_set[image_idx].first->GetPixel(x, y);
                red_av[7][7] += pixel.Red;
                green_av[7][7] += pixel.Green;
                blue_av[7][7] += pixel.Blue;
            }
        }
        red_av[7][7] /= square4;
        green_av[7][7] /= square4;
        blue_av[7][7] /= square4;
        //the end of counting average colours

        vector <float> colors;
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                colors.push_back(red_av[i][j]/255);
                colors.push_back(green_av[i][j]/255);
                colors.push_back(blue_av[i][j]/255);
            }
        }

        HOG(data_set[image_idx].first, one_image_features);

        BMP *first_quater = new BMP();
        BMP *second_quater = new BMP();
        BMP *third_quater = new BMP();
        BMP *fourth_quater = new BMP();

        first_quater->SetSize(w/2, h/2);
        second_quater->SetSize(w/2 + w%2, h/2);
        third_quater->SetSize(w/2 + w%2, h/2 + h%2);
        fourth_quater->SetSize(w/2, h/2 + h%2);

        //dividing image on 4 parts and counting the HOG descriptor for each part
        for(uint i = 0; i < w; i++){
            for(uint j = 0; j < h; j++){
                pixel = data_set[image_idx].first->GetPixel(i, j);
                if(j < h / 2){
                    if(i < w / 2) first_quater->SetPixel(i, j, pixel);
                    else second_quater->SetPixel(i - w/2, j, pixel);
                }
                else{
                    if(i < w / 2) fourth_quater->SetPixel(i, j - h/2, pixel);
                    else third_quater->SetPixel(i - w/2, j - h/2, pixel);
                }
            }
        }

        HOG(first_quater, one_image_features);
        HOG(second_quater, one_image_features);
        HOG(third_quater, one_image_features);
        HOG(fourth_quater, one_image_features);

        //push in vector of features colour features
        for(int i = 0; i < 192; i++) one_image_features.push_back(colors[i]);

        //make nonlinear cernel
        vector <float> after_nonlinear_celnel;
        for(uint i = 0; i < one_image_features.size(); i++){
            for(int n = -1; n <= 1; n++){
                if(one_image_features[i] <= 0.0){
                    after_nonlinear_celnel.push_back(0);
                    after_nonlinear_celnel.push_back(0);
                }
                else{
                    after_nonlinear_celnel.push_back(cos(-1 * n * 0.5 * log(one_image_features[i])) * sqrt(one_image_features[i] / cosh(pi * n * 0.5)));
                    after_nonlinear_celnel.push_back(sin(-1 * n * 0.5 * log(one_image_features[i])) * sqrt(one_image_features[i] / cosh(pi * n * 0.5)));
                }
            }
        }
        features->push_back(make_pair(after_nonlinear_celnel, data_set[image_idx].second));
        delete fourth_quater;
        delete third_quater;
        delete second_quater;
        delete first_quater;
    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");

        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
