/* 

 * Copyright (c) 2012-2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <ctype.h>

using namespace cv;
using namespace std;

const bool showImages = false;
const char* windowName = "LK";

const int MAX_COUNT = 100;

static void help()
{
    cout <<
            "\n"
            "This is a test data generator for Lukas-Kanade optical flow test,\n"
            "Using OpenCV version " CV_VERSION "\n"
            "Run generator from \"test_data\" directory\n";
}

void generate_data(Size winSize, String image0, String image1, String suffix = String(""))
{
    cout << winSize << endl;

    TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

    Mat frames[2] = {
            imread(image0, CV_LOAD_IMAGE_GRAYSCALE),
            imread(image1, CV_LOAD_IMAGE_GRAYSCALE)
    };

    ofstream res;
    res.open(cv::format("optflow_pyrlk%s_%dx%d.txt", suffix.c_str(), winSize.width, winSize.height).c_str());

    if (showImages)
        namedWindow(windowName, 1);

    Mat prevImage, image, imageShow;
    vector<Point2f> points[2];

    frames[0].copyTo(image);

    image.copyTo(imageShow);
    // initialization
    goodFeaturesToTrack(image, points[0], MAX_COUNT, 0.01, 10, Mat());
    if (showImages)
    {
        for (size_t i = 0; i < points[0].size(); i++)
        {
            circle(imageShow, points[0][i], 3, Scalar(255, 255, 255), -1, 8);
        }
        imshow(windowName, imageShow);
        waitKey(0);
    }

    cv::swap(prevImage, image);
    frames[1].copyTo(image);

    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prevImage, image, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
    size_t i, k;
    for (i = k = 0; i < points[1].size(); i++)
    {
        res << i << " " << (int)status[i] << " " << points[0][i].x << " " << points[0][i].y << " " << points[1][i].x << " " << points[1][i].y << endl;
        if (showImages)
            circle(imageShow, points[0][i], 3, Scalar(255, 255, 255), -1, 8);

        if (!status[i])
            continue;

        if (showImages)
            circle(imageShow, points[1][i], 3, Scalar(0, 0, 0), -1, 8);
    }
    if (showImages)
    {
        imshow(windowName, imageShow);
        waitKey(0);
    }
}

int main(int argc, char** argv)
{
    help();
    generate_data(Size(5, 5), "optflow_00.bmp", "optflow_01.bmp");
    generate_data(Size(9, 9), "optflow_00.bmp", "optflow_01.bmp");
}
