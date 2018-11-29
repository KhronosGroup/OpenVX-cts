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

#include <stdio.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

void generate_reference_result(const char* src_name, int winsz, int low_thresh, int high_thresh, bool use_l2)
{
    Mat src = imread(src_name, IMREAD_GRAYSCALE);
    Mat dst;
    if (src.empty())
    {
        printf("failed to open %s\n", src_name);
        exit(-1);
    }

    Canny(src, dst, low_thresh, high_thresh, winsz, use_l2);

    //2-pixel borders:
    rectangle(dst, Point(0,0), Point(dst.cols-1, dst.rows-1), 255, 2, 4, 0);

    char buff[1024];
    sprintf(buff, "canny_%dx%d_%d_%d_%s_%s", winsz, winsz, low_thresh, high_thresh, use_l2 ? "L2" : "L1", src_name);
    imwrite(buff, dst);
}

int main(int, char**)
{
    generate_reference_result("lena_gray.bmp", 3, 70,   71, false);
    generate_reference_result("lena_gray.bmp", 3, 70,   71, true);
    generate_reference_result("lena_gray.bmp", 3, 90,  130, false);
    generate_reference_result("lena_gray.bmp", 3, 90,  130, true);
    generate_reference_result("lena_gray.bmp", 3, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 3, 100, 120, true);
    generate_reference_result("lena_gray.bmp", 3, 120, 120, false);
    generate_reference_result("lena_gray.bmp", 3, 150, 220, false);
    generate_reference_result("lena_gray.bmp", 3, 150, 220, true);
    generate_reference_result("lena_gray.bmp", 5, 100, 100, false);
    generate_reference_result("lena_gray.bmp", 5, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 5, 100, 120, true);
    generate_reference_result("lena_gray.bmp", 7, 80,   80, false);
    generate_reference_result("lena_gray.bmp", 7, 100, 120, false);
    generate_reference_result("lena_gray.bmp", 7, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 7, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 5, 100, 120, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 150, 220, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 70,   71, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 70,   71, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 90,  125, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 90,  130, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 100, 120, false);
    generate_reference_result("blurred_lena_gray.bmp", 3, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 3, 150, 220, true);
    generate_reference_result("blurred_lena_gray.bmp", 5, 100, 120, true);
    generate_reference_result("blurred_lena_gray.bmp", 7, 100, 120, false);

    generate_reference_result("lena_gray.bmp", 5, 1200, 1440, false);
    generate_reference_result("lena_gray.bmp", 5, 1200, 1440, true);
    generate_reference_result("lena_gray.bmp", 7, 16000, 19200, false);
    generate_reference_result("lena_gray.bmp", 7, 16000, 19200, true);
    return 0;
}
