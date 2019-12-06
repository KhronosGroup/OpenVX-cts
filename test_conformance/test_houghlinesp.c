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

#ifdef OPENVX_USE_ENHANCED_VISION

#include "test_engine/test.h"

#include <stdint.h>
#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <math.h>

#define MAX_NUM_EXP_LINES 100

TESTCASE(Houghlinesp, CT_VXContext, ct_setup_vx_context, 0)

TEST(Houghlinesp, testNodeCreation)
{
    vx_context context = context_->vx_context_;
    vx_image input = 0;
    vx_uint32 src_width;
    vx_uint32 src_height;
    vx_array lines_array = 0;
    vx_scalar num_lines = 0;
    vx_uint32 numlines = 0;
    vx_hough_lines_p_t param_hough_lines = {1, M_PI/180, 50, 50, 10, M_PI, 0};

    vx_graph graph = 0;
    vx_node node = 0;

    src_width = 640;
    src_height = 320;

    ASSERT_VX_OBJECT(input = vxCreateImage(context, src_width, src_height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(lines_array = vxCreateArray(context, VX_TYPE_LINE_2D, src_width * src_height * sizeof(vx_uint32)), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(num_lines = vxCreateScalar(context, VX_TYPE_SIZE, &numlines), VX_TYPE_SCALAR);

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxHoughLinesPNode(graph, input, &param_hough_lines, lines_array, num_lines), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseArray(&lines_array));
    VX_CALL(vxReleaseScalar(&num_lines));
    VX_CALL(vxReleaseImage(&input));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(lines_array == 0);
    ASSERT(num_lines == 0);
    ASSERT(input == 0);
}

static CT_Image hough_lines_read_image(const char *fileName, int width, int height, vx_df_image format)
{
    CT_Image image_load = NULL, image_ret = NULL;
    ASSERT_(return 0, width == 0 && height == 0);
    ASSERT_(return 0, format == VX_DF_IMAGE_U1 || format == VX_DF_IMAGE_U8);

    image_load = ct_read_image(fileName, 1);
    ASSERT_(return 0, image_load);
    ASSERT_(return 0, image_load->format == VX_DF_IMAGE_U8);

    if (format == VX_DF_IMAGE_U1)
    {
        ASSERT_NO_FAILURE_(return 0, threshold_U8_ct_image(image_load, 127));   // Threshold to make the U1 image less trivial
        ASSERT_NO_FAILURE_(return 0, image_ret = ct_allocate_image(image_load->width, image_load->height, VX_DF_IMAGE_U1));
        ASSERT_NO_FAILURE_(return 0, U8_ct_image_to_U1_ct_image(image_load, image_ret));
    }
    else
    {
        image_ret = image_load;
    }

    ASSERT_(return 0, image_ret);
    ASSERT_(return 0, image_ret->format == format);

    return image_ret;
}

static vx_bool similar_lines(vx_line2d_t act, vx_line2d_t exp, vx_float32 eps)
{
    #define LINE_RATIO 1.2

    if (fabs(act.start_x - exp.start_x) <= eps && fabs(act.start_y - exp.start_y) <= eps &&
        fabs(act.end_x - exp.end_x) <= eps && fabs(act.end_y - exp.end_y) <= eps)
    {
        return vx_true_e;
    }
    else
    {
        vx_float32 ax = fabs(act.start_x - act.end_x);
        vx_float32 ay = fabs(act.end_y - act.start_y);
        vx_float32 gx = fabs(exp.end_x - exp.start_x);
        vx_float32 gy = fabs(exp.end_y - exp.start_y);
        vx_float32 sx = fabs(act.start_x - exp.start_x);
        vx_float32 sy = fabs(act.start_y - exp.start_y);
        vx_float32 ex = fabs(act.end_x - exp.end_x);
        vx_float32 ey = fabs(act.end_y - exp.end_y);
        vx_float32 dg = sqrt(gx * gx + gy * gy);
        vx_float32 ds = sqrt(sx * sx + sy * sy);
        vx_float32 de = sqrt(ex * ex + ey * ey);
        if (gx != 0 && ax != 0)
        {
            if (fabs(gy / gx - ay / ax) <= 0.1 && fabs(gy / gx - sy / sx) <= 0.1 && fabs(gy / gx - ey / ex) <= 0.1
                && (ds < (LINE_RATIO * dg)) && (de < (LINE_RATIO * dg)))
                return vx_true_e;
        }
        else if (gy != 0 && ay != 0)
        {
            if (fabs(gx / gy - ax / ay) <= 0.1 && fabs(gx / gy - sx / sy) <= 0.1 && fabs(gx / gy - ex / ey) <= 0.1
                && (ds < (LINE_RATIO * dg)) && (de < (LINE_RATIO * dg)))
                return vx_true_e;
        }

        return vx_false_e;
    }
}

static vx_status countLine2dIntersection(const vx_line2d_t *expect_lines, const vx_line2d_t *actual_lines, vx_int32 exp_lines_num, vx_int32 actual_lines_num, vx_float32 eps)
{
    vx_status status = VX_FAILURE;
    vx_int32 count = 0;
    if (exp_lines_num && actual_lines_num)
    {
        for (vx_int32 x = 0; x < actual_lines_num; x++)
        {
            vx_line2d_t act = actual_lines[x];
            for (vx_int32 y = 0; y < exp_lines_num; y++)
            {
                vx_line2d_t exp = expect_lines[y];
                vx_bool sim = similar_lines(act, exp, eps);
                if (sim)
                {
                    count++;
                    break;
                }
            }
        }
    }
    if ((vx_float32)count / (exp_lines_num < actual_lines_num ? exp_lines_num : actual_lines_num) >= 0.8)
        status = VX_SUCCESS;
    return status;
}

static vx_status houghlinesp_check(vx_array lines_array, vx_scalar num_lines, const char* result_filename)
{
    vx_status status = VX_FAILURE;
    vx_size sz = 0;
    void* buf = 0;

    char file[MAXPATHLENGTH];
    sz = snprintf(file, MAXPATHLENGTH, "%s/%s", ct_get_test_file_path(), result_filename);
    FILE* f = fopen(file, "rb");
    ASSERT_(return VX_FAILURE, f);
    fseek(f, 0, SEEK_END);

    sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    ASSERT_(return VX_FAILURE, buf = ct_alloc_mem(sz + 1));
    ASSERT_(return VX_FAILURE, sz == fread(buf, 1, sz, f));
    fclose(f); f = NULL;
    ((vx_int8*)buf)[sz] = 0;

    vx_size lines_array_stride = 0;
    void *lines_array_ptr = NULL;
    vx_map_id lines_array_map_id;
    vx_size lines_array_length;
    vxQueryArray(lines_array, VX_ARRAY_NUMITEMS, &lines_array_length, sizeof(lines_array_length));
    vxMapArrayRange(lines_array, 0, lines_array_length, &lines_array_map_id, &lines_array_stride, &lines_array_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    vx_line2d_t *lines_array_p = (vx_line2d_t *)lines_array_ptr;
    vx_line2d_t *exp_lines = 0;
    ASSERT_(return VX_FAILURE, exp_lines = (vx_line2d_t *)ct_alloc_mem(sizeof(vx_line2d_t) * MAX_NUM_EXP_LINES));

    vx_int32 id = 0;
    char * pos = (char *)buf;
    char * next = 0;
    while (pos && (next = strchr(pos, '\n')))
    {
        vx_float32 x1, y1, x2, y2;
        vx_int32 line_id;

        *next = 0;
        (void)sscanf(pos, "%d %f %f %f %f", &line_id, &x1, &y1, &x2, &y2);
        exp_lines[id].start_x = x1;
        exp_lines[id].start_y = y1;
        exp_lines[id].end_x = x2;
        exp_lines[id].end_y = y2;
        pos = next + 1;
        id++;
    }
    vx_int32 exp_lines_num = id + 1;
    status = countLine2dIntersection(exp_lines, lines_array_p, exp_lines_num, lines_array_length, 2.0f);
    vxUnmapArrayRange(lines_array, lines_array_map_id);
    return status;
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    vx_hough_lines_p_t param_hough_lines;
    const char* result_filename;
    vx_df_image format;
} Arg;

#define PARAMETERS \
    ARG("case1_1_180_50_50_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U8), \
    ARG("case1_1_170_40_40_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/170, 40, 40, 10, M_PI, 0}, "hough_lines_1_170_40_40_10.txt", VX_DF_IMAGE_U8), \
    ARG("case1_1_180_40_40_9_HoughLines",  hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 40, 40, 9,  M_PI, 0}, "hough_lines_1_180_40_40_9.txt",  VX_DF_IMAGE_U8), \
    ARG("case1_2_180_50_50_9_HoughLines",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U8), \
    ARG("case1_1_190_40_40_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/190, 40, 40, 10, M_PI, 0}, "hough_lines_1_190_40_40_10.txt", VX_DF_IMAGE_U8), \
    \
    ARG("_U1_/case1_1_180_50_50_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U1), \
    ARG("_U1_/case1_1_170_40_40_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/170, 40, 40, 10, M_PI, 0}, "hough_lines_1_170_40_40_10.txt", VX_DF_IMAGE_U1), \
    ARG("_U1_/case1_1_180_40_40_9_HoughLines",  hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 40, 40, 9,  M_PI, 0}, "hough_lines_1_180_40_40_9.txt",  VX_DF_IMAGE_U1), \
    ARG("_U1_/case1_2_180_50_50_9_HoughLines",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U1), \
    ARG("_U1_/case1_1_190_40_40_10_HoughLines", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/190, 40, 40, 10, M_PI, 0}, "hough_lines_1_190_40_40_10.txt", VX_DF_IMAGE_U1) \

TEST_WITH_ARG(Houghlinesp, testGraphProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_array lines_array = 0;
    vx_scalar num_lines = 0;
    vx_uint32 numlines = 0;
    CT_Image src = NULL;
    vx_hough_lines_p_t param_lines_p = arg_->param_hough_lines;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0, arg_->format));

    src_width = src->width;
    src_height = src->height;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(lines_array = vxCreateArray(context, VX_TYPE_LINE_2D, src_width * src_height * sizeof(vx_line2d_t)), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(num_lines = vxCreateScalar(context, VX_TYPE_SIZE, &numlines), VX_TYPE_SCALAR);
    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);
    ASSERT_VX_OBJECT(node = vxHoughLinesPNode(graph, src_image, &param_lines_p, lines_array, num_lines), VX_TYPE_NODE);

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    ASSERT_NO_FAILURE(status = houghlinesp_check(lines_array, num_lines, arg_->result_filename));
    ASSERT(status == VX_SUCCESS);

    VX_CALL(vxReleaseNode(&node));
    VX_CALL(vxReleaseGraph(&graph));
    VX_CALL(vxReleaseArray(&lines_array));
    VX_CALL(vxReleaseScalar(&num_lines));

    ASSERT(node == 0);
    ASSERT(graph == 0);
    ASSERT(lines_array == 0);
    ASSERT(num_lines == 0);

    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(src_image == 0);
}

TEST_WITH_ARG(Houghlinesp, testImmediateProcessing, Arg,
    PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;

    CT_Image src = NULL;

    vx_array lines_array = 0;
    vx_scalar num_lines = 0;
    vx_uint32 numlines = 0;

    vx_hough_lines_p_t param_lines_p = arg_->param_hough_lines;
    vx_status status;

    vx_uint32 src_width;
    vx_uint32 src_height;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0, arg_->format));

    src_width = src->width;
    src_height = src->height;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(lines_array = vxCreateArray(context, VX_TYPE_LINE_2D, src_width * src_height * sizeof(vx_line2d_t)), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(num_lines = vxCreateScalar(context, VX_TYPE_SIZE, &numlines), VX_TYPE_SCALAR);

    VX_CALL(vxuHoughLinesP(context, src_image, &param_lines_p, lines_array, num_lines));

    ASSERT_NO_FAILURE(status = houghlinesp_check(lines_array, num_lines, arg_->result_filename));
    ASSERT(status == VX_SUCCESS);

    VX_CALL(vxReleaseArray(&lines_array));
    VX_CALL(vxReleaseScalar(&num_lines));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(lines_array == 0);
    ASSERT(num_lines == 0);
    ASSERT(src_image == 0);
}

typedef struct {
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    vx_hough_lines_p_t param_hough_lines;
    const char* result_filename;
    vx_df_image format;
    vx_rectangle_t region_shift;
} ValidRegionTest_Arg;

#define REGION_PARAMETERS \
    ARG("case1_1_180_50_50_10_HoughLines_RegionShrink=1", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U8, {1, 1, -1, -1}), \
    ARG("case1_2_180_50_50_9_HoughLines_RegionShrink=1",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U8, {1, 1, -1, -1}), \
    ARG("case1_1_180_50_50_10_HoughLines_RegionShrink=7", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U8, {7, 7, -7, -7}), \
    ARG("case1_2_180_50_50_9_HoughLines_RegionShrink=7",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U8, {7, 7, -7, -7}), \
    \
    ARG("_U1_/case1_1_180_50_50_10_HoughLines_RegionShrink=1", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U1, {1, 1, -1, -1}), \
    ARG("_U1_/case1_2_180_50_50_9_HoughLines_RegionShrink=1",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U1, {1, 1, -1, -1}), \
    ARG("_U1_/case1_1_180_50_50_10_HoughLines_RegionShrink=7", hough_lines_read_image, "hough_lines.bmp", {1, M_PI/180, 50, 50, 10, M_PI, 0}, "hough_lines_1_180_50_50_10.txt", VX_DF_IMAGE_U1, {7, 7, -7, -7}), \
    ARG("_U1_/case1_2_180_50_50_9_HoughLines_RegionShrink=7",  hough_lines_read_image, "hough_lines.bmp", {2, M_PI/180, 50, 50, 9,  M_PI, 0}, "hough_lines_2_180_50_50_9.txt",  VX_DF_IMAGE_U1, {7, 7, -7, -7}) \

// For small valid region shrinks (like the ones in these tests) the already existing reference output line lists
// still apply because the objects in hough_lines.bmp are fairly centered in the image and far from the edges
TEST_WITH_ARG(Houghlinesp, testWithValidRegion, ValidRegionTest_Arg,
    REGION_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0;

    CT_Image src = NULL;
    vx_uint32 src_width;
    vx_uint32 src_height;

    vx_array lines_array = 0;
    vx_scalar num_lines = 0;
    vx_uint32 numlines = 0;

    vx_hough_lines_p_t param_lines_p = arg_->param_hough_lines;
    vx_status status;

    vx_rectangle_t rect = {0, 0, 0, 0}, rect_shft = arg_->region_shift;

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, 0, 0, arg_->format));

    src_width = src->width;
    src_height = src->height;

    ASSERT_VX_OBJECT(src_image = ct_image_to_vx_image(src, context), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(lines_array = vxCreateArray(context, VX_TYPE_LINE_2D, src_width * src_height * sizeof(vx_line2d_t)), VX_TYPE_ARRAY);
    ASSERT_VX_OBJECT(num_lines = vxCreateScalar(context, VX_TYPE_SIZE, &numlines), VX_TYPE_SCALAR);

    ASSERT_NO_FAILURE(vxGetValidRegionImage(src_image, &rect));
    ALTERRECTANGLE(rect, rect_shft.start_x, rect_shft.start_y, rect_shft.end_x, rect_shft.end_y);
    ASSERT_NO_FAILURE(vxSetImageValidRectangle(src_image, &rect));

    VX_CALL(vxuHoughLinesP(context, src_image, &param_lines_p, lines_array, num_lines));

    ASSERT_NO_FAILURE(status = houghlinesp_check(lines_array, num_lines, arg_->result_filename));
    ASSERT(status == VX_SUCCESS);

    VX_CALL(vxReleaseArray(&lines_array));
    VX_CALL(vxReleaseScalar(&num_lines));
    VX_CALL(vxReleaseImage(&src_image));

    ASSERT(lines_array == 0);
    ASSERT(num_lines == 0);
    ASSERT(src_image == 0);
}

TESTCASE_TESTS(Houghlinesp,
               testNodeCreation,
               testGraphProcessing,
               testImmediateProcessing,
               testWithValidRegion)

#endif //OPENVX_USE_ENHANCED_VISION
