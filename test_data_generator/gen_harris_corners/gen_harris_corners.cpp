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
#include <assert.h>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Params
{
    double minDistance;
    double k;
    int ksize;
    int blockSize;
};

static const Params g_cparams[] =
{
    { 0.0, 0.04, 3, 3 }, { 0.0, 0.04, 3, 5 }, { 0.0, 0.04, 3, 7 },
    { 0.0, 0.04, 5, 3 }, { 0.0, 0.04, 5, 5 }, { 0.0, 0.04, 5, 7 },
    { 0.0, 0.04, 7, 3 }, { 0.0, 0.04, 7, 5 }, { 0.0, 0.04, 7, 7 },

    { 0.0, 0.10, 3, 3 }, { 0.0, 0.10, 3, 5 }, { 0.0, 0.10, 3, 7 },
    { 0.0, 0.10, 5, 3 }, { 0.0, 0.10, 5, 5 }, { 0.0, 0.10, 5, 7 },
    { 0.0, 0.10, 7, 3 }, { 0.0, 0.10, 7, 5 }, { 0.0, 0.10, 7, 7 },

    { 0.0, 0.15, 3, 3 }, { 0.0, 0.15, 3, 5 }, { 0.0, 0.15, 3, 7 },
    { 0.0, 0.15, 5, 3 }, { 0.0, 0.15, 5, 5 }, { 0.0, 0.15, 5, 7 },
    { 0.0, 0.15, 7, 3 }, { 0.0, 0.15, 7, 5 }, { 0.0, 0.15, 7, 7 },


    {3.0, 0.04, 3, 3}, {3.0, 0.04, 3, 5}, {3.0, 0.04, 3, 7},
    {3.0, 0.04, 5, 3}, {3.0, 0.04, 5, 5}, {3.0, 0.04, 5, 7},
    {3.0, 0.04, 7, 3}, {3.0, 0.04, 7, 5}, {3.0, 0.04, 7, 7},

    {3.0, 0.10, 3, 3}, {3.0, 0.10, 3, 5}, {3.0, 0.10, 3, 7},
    {3.0, 0.10, 5, 3}, {3.0, 0.10, 5, 5}, {3.0, 0.10, 5, 7},
    {3.0, 0.10, 7, 3}, {3.0, 0.10, 7, 5}, {3.0, 0.10, 7, 7},

    {3.0, 0.15, 3, 3}, {3.0, 0.15, 3, 5}, {3.0, 0.15, 3, 7},
    {3.0, 0.15, 5, 3}, {3.0, 0.15, 5, 5}, {3.0, 0.15, 5, 7},
    {3.0, 0.15, 7, 3}, {3.0, 0.15, 7, 5}, {3.0, 0.15, 7, 7},


    {5.0, 0.04, 3, 3}, {5.0, 0.04, 3, 5}, {5.0, 0.04, 3, 7},
    {5.0, 0.04, 5, 3}, {5.0, 0.04, 5, 5}, {5.0, 0.04, 5, 7},
    {5.0, 0.04, 7, 3}, {5.0, 0.04, 7, 5}, {5.0, 0.04, 7, 7},

    {5.0, 0.10, 3, 3}, {5.0, 0.10, 3, 5}, {5.0, 0.10, 3, 7},
    {5.0, 0.10, 5, 3}, {5.0, 0.10, 5, 5}, {5.0, 0.10, 5, 7},
    {5.0, 0.10, 7, 3}, {5.0, 0.10, 7, 5}, {5.0, 0.10, 7, 7},

    {5.0, 0.15, 3, 3}, {5.0, 0.15, 3, 5}, {5.0, 0.15, 3, 7},
    {5.0, 0.15, 5, 3}, {5.0, 0.15, 5, 5}, {5.0, 0.15, 5, 7},
    {5.0, 0.15, 7, 3}, {5.0, 0.15, 7, 5}, {5.0, 0.15, 7, 7},


    { 30.0, 0.04, 3, 3 }, { 30.0, 0.04, 3, 5 }, { 30.0, 0.04, 3, 7 },
    { 30.0, 0.04, 5, 3 }, { 30.0, 0.04, 5, 5 }, { 30.0, 0.04, 5, 7 },
    { 30.0, 0.04, 7, 3 }, { 30.0, 0.04, 7, 5 }, { 30.0, 0.04, 7, 7 },

    { 30.0, 0.10, 3, 3 }, { 30.0, 0.10, 3, 5 }, { 30.0, 0.10, 3, 7 },
    { 30.0, 0.10, 5, 3 }, { 30.0, 0.10, 5, 5 }, { 30.0, 0.10, 5, 7 },
    { 30.0, 0.10, 7, 3 }, { 30.0, 0.10, 7, 5 }, { 30.0, 0.10, 7, 7 },

    { 30.0, 0.15, 3, 3 }, { 30.0, 0.15, 3, 5 }, { 30.0, 0.15, 3, 7 },
    { 30.0, 0.15, 5, 3 }, { 30.0, 0.15, 5, 5 }, { 30.0, 0.15, 5, 7 },
    { 30.0, 0.15, 7, 3 }, { 30.0, 0.15, 7, 5 }, { 30.0, 0.15, 7, 7 },
};

static void generateHarrisCornerDataSingle(const char * filepath, const char *outprefix, double minDistance, double k, int ksize, int blockSize)
{
    const double qualityLevel = 0.05;
    cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        printf("failed to open %s\n", filepath);
        exit(-1);
    }

    char outfilename[2048];
    sprintf(outfilename, "%s_%0.2f_%0.2f_%d_%d.txt", outprefix, minDistance, k, ksize, blockSize);
    cv::Mat corners;
    std::ofstream stream(outfilename);
    cv::goodFeaturesToTrack(image, corners, image.cols * image.rows, qualityLevel, minDistance, cv::noArray(), blockSize, true, k, ksize);
    float scale = (1 << (ksize - 1)) * blockSize * 255.f;
    scale = scale * scale * scale * scale;
    stream << corners.rows << std::endl;
    for (int i = 0; i < corners.rows; i++)
    {
        cv::Point3f *pt = (cv::Point3f *)corners.ptr(i);
        if ((0 <= pt->x) && (pt->x < image.cols) && (0 <= pt->y) && (pt->y < image.rows))
            stream << pt->x << " " << pt->y << " " << pt->z * scale << std::endl;
    }
}

static void generateHarrisCornerDataSuite(const char * filepath, const char *outprefix)
{
    size_t params_count = sizeof(g_cparams) / sizeof(Params);
    for (size_t i = 0; i < params_count; i++)
    {
        generateHarrisCornerDataSingle(filepath, outprefix, g_cparams[i].minDistance, g_cparams[i].k, g_cparams[i].ksize, g_cparams[i].blockSize);
    }
}

int main(int argc, char* argv[])
{
    generateHarrisCornerDataSuite("harriscorners/hc_fsc.bmp", "harriscorners/hc_fsc");
    generateHarrisCornerDataSuite("harriscorners/hc_msc.bmp", "harriscorners/hc_msc");
    return 0;
}
