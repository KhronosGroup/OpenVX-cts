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

#if defined OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

/* ***************************************************************************
//  local auxiliary function definitions (only those that need early definitions)
*/
static void mem_free(void**ptr);

static CT_Image own_generate_rand_image(const char* fileName, int width, int height, vx_df_image format);

static vx_uint32 own_plane_subsampling_x(vx_df_image format, vx_uint32 plane);

static vx_uint32 own_plane_subsampling_y(vx_df_image format, vx_uint32 plane);

static void own_allocate_image_ptrs(vx_df_image format, int width, int height, vx_uint32* nplanes, void* ptrs[],
    vx_imagepatch_addressing_t addr[], vx_pixel_value_t* val);


/* ***************************************************************************
//  Image tests
*/
TESTCASE(Image, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* name;
    vx_df_image format;
} ImageFormat_Arg;

typedef struct
{
    const char* name;
    int width;
    int height;
    vx_df_image format;
} ImageDims_Arg;

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width;
    int height;
    vx_df_image format;
    vx_bool have_roi;
} ImageGenerator_Arg;

#define VX_PLANE_MAX (4)

#define IMAGE_FORMAT_PARAMETERS_BASELINE \
    ARG_ENUM(VX_DF_IMAGE_U8), \
    ARG_ENUM(VX_DF_IMAGE_U16), \
    ARG_ENUM(VX_DF_IMAGE_S16), \
    ARG_ENUM(VX_DF_IMAGE_U32), \
    ARG_ENUM(VX_DF_IMAGE_S32), \
    ARG_ENUM(VX_DF_IMAGE_RGB), \
    ARG_ENUM(VX_DF_IMAGE_RGBX), \
    ARG_ENUM(VX_DF_IMAGE_NV12), \
    ARG_ENUM(VX_DF_IMAGE_NV21), \
    ARG_ENUM(VX_DF_IMAGE_UYVY), \
    ARG_ENUM(VX_DF_IMAGE_YUYV), \
    ARG_ENUM(VX_DF_IMAGE_IYUV), \
    ARG_ENUM(VX_DF_IMAGE_YUV4)

#define ADD_IMAGE_FORMATS(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U8", __VA_ARGS__, VX_DF_IMAGE_U8)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U16", __VA_ARGS__, VX_DF_IMAGE_U16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S16", __VA_ARGS__, VX_DF_IMAGE_S16)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U32", __VA_ARGS__, VX_DF_IMAGE_U32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_S32", __VA_ARGS__, VX_DF_IMAGE_S32)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGB", __VA_ARGS__, VX_DF_IMAGE_RGB)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_RGBX", __VA_ARGS__, VX_DF_IMAGE_RGBX)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_UYVY", __VA_ARGS__, VX_DF_IMAGE_UYVY)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUYV", __VA_ARGS__, VX_DF_IMAGE_YUYV)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV12", __VA_ARGS__, VX_DF_IMAGE_NV12)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV21", __VA_ARGS__, VX_DF_IMAGE_NV21)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUV4", __VA_ARGS__, VX_DF_IMAGE_YUV4)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_IYUV", __VA_ARGS__, VX_DF_IMAGE_IYUV))

#define ADD_IMAGE_FORMAT_U1(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_U1", __VA_ARGS__, VX_DF_IMAGE_U1))

#define ADD_IMAGE_ROI(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/ROI=true", __VA_ARGS__, vx_true_e)), \
    CT_EXPAND(nextmacro(testArgName "/ROI=false", __VA_ARGS__, vx_false_e))

#define NO_IMAGE_ROI(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "", __VA_ARGS__, vx_false_e))

#define TEST_IMAGE_RANDOM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS(     "rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   NO_IMAGE_ROI, ARG, own_generate_rand_image, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, NO_IMAGE_ROI, ARG, own_generate_rand_image, NULL)

#define TEST_IMAGE_RANDOM_IMAGE_WITH_ROI_PARAMETERS \
    CT_GENERATE_PARAMETERS(     "rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   ADD_IMAGE_ROI, ARG, own_generate_rand_image, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, ADD_IMAGE_ROI, ARG, own_generate_rand_image, NULL)

TEST_WITH_ARG(Image, testRngImageCreation, ImageFormat_Arg,
    IMAGE_FORMAT_PARAMETERS_BASELINE,
    ARG_ENUM(VX_DF_IMAGE_VIRT),
    ARG("_U1_/VX_DF_IMAGE_U1", VX_DF_IMAGE_U1),
)
{
    vx_context  context = context_->vx_context_;
    vx_image    image   = 0;
    vx_image    clone   = 0;
    vx_df_image format  = arg_->format;

    image = vxCreateImage(context, 8, 8, format);

    if (format == VX_DF_IMAGE_VIRT)
    {
        ASSERT_NE_VX_STATUS(VX_SUCCESS, vxGetStatus((vx_reference)image));
        PASS();
    }

    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    ct_fill_image_random(image, &CT()->seed_);

    clone = ct_clone_image(image, 0);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseImage(&clone));

    ASSERT(image == 0);
    ASSERT(clone == 0);
} /* testRngImageCreation() */

/*
// Creation and destruction of U1 images should be supported even without the U1 conformance profile
*/
TEST(Image, testImageCreation_U1)
{
    // Test vxCreateImage()
    vx_context context = context_->vx_context_;
    vx_image image = 0;
    vx_uint32 width = 16;
    vx_uint32 height = 16;
    vx_df_image format = VX_DF_IMAGE_U1;

    image = vxCreateImage(context, width, height, format);

    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    VX_CALL(vxReleaseImage(&image));

    ASSERT(image == 0);
} /* testImageCreation_U1() */

TEST_WITH_ARG(Image, testVirtualImageCreation, ImageFormat_Arg,
    IMAGE_FORMAT_PARAMETERS_BASELINE,
    ARG_ENUM(VX_DF_IMAGE_VIRT),
    ARG_ENUM(VX_DF_IMAGE_U1),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_image   clone   = 0;
    vx_df_image  format  = arg_->format;

    vx_graph graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    image = vxCreateVirtualImage(graph, 4, 4, format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    clone = ct_clone_image(image, graph);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseImage(&clone));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(image == 0);
    ASSERT(clone == 0);
    ASSERT(graph == 0);
} /* testVirtualImageCreation() */

TEST_WITH_ARG(Image, testVirtualImageCreationDims, ImageDims_Arg,
    ARG("0_0_REAL", 0, 0, VX_DF_IMAGE_U8),
    ARG("DISABLED_0_4_REAL", 0, 4, VX_DF_IMAGE_U8),
    ARG("DISABLED_4_0_REAL", 4, 0, VX_DF_IMAGE_U8),
    ARG("4_4_REAL", 4, 4, VX_DF_IMAGE_U8),
    ARG("0_0_VIRT", 0, 0, VX_DF_IMAGE_VIRT),
    ARG("DISABLED_0_4_VIRT", 0, 4, VX_DF_IMAGE_VIRT),
    ARG("DISABLED_4_0_VIRT", 4, 0, VX_DF_IMAGE_VIRT),
    ARG("4_4_VIRT", 4, 4, VX_DF_IMAGE_VIRT),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_image   clone   = 0;

    vx_graph graph = vxCreateGraph(context);
    ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);

    image = vxCreateVirtualImage(graph, arg_->width, arg_->height, arg_->format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    clone = ct_clone_image(image, graph);
    ASSERT_VX_OBJECT(clone, VX_TYPE_IMAGE);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseImage(&clone));
    VX_CALL(vxReleaseGraph(&graph));

    ASSERT(image == 0);
    ASSERT(clone == 0);
    ASSERT(graph == 0);
} /* testVirtualImageCreationDims() */

TEST_WITH_ARG(Image, testCreateImageFromHandle, ImageGenerator_Arg,
    TEST_IMAGE_RANDOM_IMAGE_PARAMETERS
)
{
    vx_uint32 n;
    vx_uint32 nplanes;
    vx_context context = context_->vx_context_;
    vx_image image = 0;
    vx_imagepatch_addressing_t addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };

    int channel[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    CT_Image src = NULL;
    CT_Image tst = NULL;

    void* ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    switch (arg_->format)
    {
    case VX_DF_IMAGE_U1:
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
        channel[0] = VX_CHANNEL_0;
        break;

    case VX_DF_IMAGE_RGB:
    case VX_DF_IMAGE_RGBX:
        channel[0] = VX_CHANNEL_R;
        channel[1] = VX_CHANNEL_G;
        channel[2] = VX_CHANNEL_B;
        channel[3] = VX_CHANNEL_A;
        break;

    case VX_DF_IMAGE_UYVY:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV:
        channel[0] = VX_CHANNEL_Y;
        channel[1] = VX_CHANNEL_U;
        channel[2] = VX_CHANNEL_V;
        break;

    default:
        ASSERT(0);
    }

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_NO_FAILURE(nplanes = ct_get_num_planes(arg_->format));

    for (n = 0; n < nplanes; n++)
    {
        addr[n].dim_x    = src->width / ct_image_get_channel_subsampling_x(src, channel[n]);
        addr[n].dim_y    = src->height / ct_image_get_channel_subsampling_y(src, channel[n]);
        addr[n].stride_x = ct_image_get_channel_step_x(src, channel[n]);
        addr[n].stride_y = ct_image_get_channel_step_y(src, channel[n]);
        if (arg_->format == VX_DF_IMAGE_U1)
            addr[n].stride_x_bits = 1;

        ptrs[n] = ct_image_get_plane_base(src, n);
    }

    EXPECT_VX_OBJECT(image = vxCreateImageFromHandle(context, arg_->format, addr, ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(tst = ct_image_from_vx_image(image));

    EXPECT_EQ_CTIMAGE(src, tst);

    VX_CALL(vxReleaseImage(&image));

    ASSERT(image == 0);
} /* testCreateImageFromHandle() */

TEST_WITH_ARG(Image, testSwapImageHandle, ImageGenerator_Arg,
    TEST_IMAGE_RANDOM_IMAGE_WITH_ROI_PARAMETERS
)
{
    vx_uint32 n;
    vx_context context = context_->vx_context_;
    vx_image image1 = 0;
    vx_image image2 = 0;
    vx_imagepatch_addressing_t addr1[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    vx_imagepatch_addressing_t addr2[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };

    vx_uint32 nplanes1 = 0;
    vx_uint32 nplanes2 = 0;
    vx_pixel_value_t val1;
    vx_pixel_value_t val2;
    vx_pixel_value_t val3;
    void* mem1_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };
    void* mem2_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };
    void* prev_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };
    CT_Image img1 = 0;
    CT_Image img2 = 0;
    CT_Image tst1 = 0;
    CT_Image tst2 = 0;

    val1.reserved[0] = 0x11;
    val1.reserved[1] = 0x22;
    val1.reserved[2] = 0x33;
    val1.reserved[3] = 0x44;

    val2.reserved[0] = 0x99;
    val2.reserved[1] = 0x88;
    val2.reserved[2] = 0x77;
    val2.reserved[3] = 0x66;

    val3.reserved[0] = 0xaa;
    val3.reserved[1] = 0xbb;
    val3.reserved[2] = 0xcc;
    val3.reserved[3] = 0xdd;

    own_allocate_image_ptrs(arg_->format, arg_->width, arg_->height, &nplanes1, mem1_ptrs, addr1, &val1);
    own_allocate_image_ptrs(arg_->format, arg_->width, arg_->height, &nplanes2, mem2_ptrs, addr2, &val2);
    EXPECT_EQ_INT(nplanes1, nplanes2);

    EXPECT_VX_OBJECT(image1 = vxCreateImageFromHandle(context, arg_->format, addr1, mem1_ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);
    EXPECT_VX_OBJECT(image2 = vxCreateImageFromHandle(context, arg_->format, addr2, mem2_ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(img1 = ct_image_from_vx_image(image1));
    ASSERT_NO_FAILURE(img2 = ct_image_from_vx_image(image2));

    if (arg_->have_roi == vx_true_e)
    {
        if(!(arg_->format == VX_DF_IMAGE_U1 && arg_->width <= 16))
        {
            vx_image roi1 = 0;
            vx_image roi2 = 0;
            vx_uint32 roi1_width;
            vx_uint32 roi1_height;

            vx_rectangle_t roi1_rect =
            {
                /* U1 subimages must start on byte boundary */
                (vx_uint32)(arg_->format == VX_DF_IMAGE_U1 ? ((arg_->width / 2 + 7) / 8) * 8 : arg_->width / 2),
                (vx_uint32)arg_->height / 2,
                (vx_uint32)arg_->width,
                (vx_uint32)arg_->height
            };

            vx_rectangle_t roi2_rect;

            /* first level subimage */
            ASSERT_VX_OBJECT(roi1 = vxCreateImageFromROI(image1, &roi1_rect), VX_TYPE_IMAGE);

            VX_CALL(vxQueryImage(roi1, VX_IMAGE_WIDTH, &roi1_width, sizeof(vx_uint32)));
            VX_CALL(vxQueryImage(roi1, VX_IMAGE_HEIGHT, &roi1_height, sizeof(vx_uint32)));

            roi2_rect.start_x = arg_->format == VX_DF_IMAGE_U1 ? ((roi1_width / 2 + 7) / 8 ) * 8 : roi1_width / 2;
            roi2_rect.start_y = roi1_height / 2;
            roi2_rect.end_x   = roi1_width;
            roi2_rect.end_y   = roi1_height;

            /* second level subimage */
            ASSERT_VX_OBJECT(roi2 = vxCreateImageFromROI(roi1, &roi2_rect), VX_TYPE_IMAGE);

            /* try to get back ROI pointers */
            ASSERT_NE_VX_STATUS(VX_SUCCESS, vxSwapImageHandle(roi2, NULL, prev_ptrs, nplanes1));

            /* try to replace and get back ROI pointers */
            ASSERT_NE_VX_STATUS(VX_SUCCESS, vxSwapImageHandle(roi2, mem2_ptrs, prev_ptrs, nplanes1));

            /* check the content of roi2 image equal image1 */
            for (n = 0; n < nplanes1; n++)
            {
                unsigned int i;
                unsigned int j;
                vx_rectangle_t rect_roi2 = { 0, 0, 0, 0 };

                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                vx_map_id map_id;

                void* plane_ptr = 0;

                VX_CALL(vxGetValidRegionImage(roi2, &rect_roi2));
                VX_CALL(vxMapImagePatch(roi2, &rect_roi2, n, &map_id, &addr, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (i = 0; i < addr.dim_y; i += addr.step_y)
                {
                    for (j = 0; j < addr.dim_x; j += addr.step_x)
                    {
                        unsigned char* p = (unsigned char*)vxFormatImagePatchAddress2d(plane_ptr, j, i, &addr);
                        if (p[0] != val1.reserved[n])
                            CT_FAIL("ROI content mismath at [x=%d, y=%d]: expected %d, actual %d", j, i, val1, p[0]);
                    }
                }
                VX_CALL(vxUnmapImagePatch(roi2, map_id));
            }

            /* replace image pointers */
            VX_CALL(vxSwapImageHandle(image1, mem2_ptrs, NULL, nplanes1));

            /* check the content of roi2 image equal image2 */
            for (n = 0; n < nplanes1; n++)
            {
                unsigned int i;
                unsigned int j;
                vx_rectangle_t rect_roi2 = { 0, 0, 0, 0 };

                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                vx_map_id map_id;

                void* plane_ptr = 0;

                VX_CALL(vxGetValidRegionImage(roi2, &rect_roi2));
                VX_CALL(vxMapImagePatch(roi2, &rect_roi2, n, &map_id, &addr, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                for (i = 0; i < addr.dim_y; i += addr.step_y)
                {
                    for (j = 0; j < addr.dim_x; j += addr.step_x)
                    {
                        unsigned char* p = (unsigned char*)vxFormatImagePatchAddress2d(plane_ptr, j, i, &addr);
                        if (p[0] != val2.reserved[n])
                            CT_FAIL("ROI content mismath at [x=%d, y=%d]: expected %d, actual %d", j, i, val2, p[0]);
                    }
                }
                VX_CALL(vxUnmapImagePatch(roi2, map_id));
            }

            /* modify the content of roi2 */
            for (n = 0; n < nplanes1; n++)
            {
                unsigned int i;
                unsigned int j;
                vx_rectangle_t rect_roi2 = { 0, 0, 0, 0 };

                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                vx_map_id map_id;

                void* plane_ptr = 0;

                VX_CALL(vxGetValidRegionImage(roi2, &rect_roi2));
                VX_CALL(vxMapImagePatch(roi2, &rect_roi2, n, &map_id, &addr, &plane_ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));
                for (i = 0; i < addr.dim_y; i += addr.step_y)
                {
                    for (j = 0; j < addr.dim_x; j += addr.step_x)
                    {
                        unsigned char* p = (unsigned char*)vxFormatImagePatchAddress2d(plane_ptr, j, i, &addr);
                        *p = val3.reserved[n];
                    }
                }
                VX_CALL(vxUnmapImagePatch(roi2, map_id));
            }

            /* reclaim image ptrs */
            VX_CALL(vxSwapImageHandle(image1, NULL, prev_ptrs, nplanes1));

            /* check that the reclaimed host memory contains the correct data */
            for (n = 0; n < nplanes2; n++)
            {
                vx_uint8* plane_ptr = (vx_uint8*)prev_ptrs[n];
                vx_uint32 i;
                vx_uint32 j;
                vx_uint32 subsampling_x = own_plane_subsampling_x(arg_->format, n);
                vx_uint32 subsampling_y = own_plane_subsampling_y(arg_->format, n);
                vx_uint32 start_x = (roi1_rect.start_x + roi2_rect.start_x) / subsampling_x;
                vx_uint32 start_y = (roi1_rect.start_y + roi2_rect.start_y) / subsampling_y;
                vx_uint32 end_x   = (vx_uint32)(arg_->width  / subsampling_x);
                vx_uint32 end_y   = (vx_uint32)(arg_->height / subsampling_y);

                for (i = 0; i < addr2[n].dim_y; i++)
                {
                    for (j = 0; j < addr2[n].dim_x; j++)
                    {
                        unsigned int k = i * addr2[n].stride_y;
                        k += (addr2[n].stride_x == 0) ? (j * addr2[n].stride_x_bits) / 8 : j * addr2[n].stride_x;

                        unsigned char p = plane_ptr[k];

                        if (i >= start_y && i <= end_y - 1 &&
                            j >= start_x && j <= end_x - 1)
                        {
                            if (p != val3.reserved[n])
                                CT_FAIL("ROI content mismath at [x=%d, y=%d]: expected %d, actual %d", j, i, val3, p);
                        }
                        else
                        {
                            if (p != val2.reserved[n])
                                CT_FAIL("ROI content mismath at [x=%d, y=%d]: expected %d, actual %d", j, i, val2, p);
                        }
                    }
                }
            }

            /* check that pointers are reclaimed in ROI */
            for (n = 0; n < nplanes1; n++)
            {
                if (prev_ptrs[n] != NULL)
                {
                    vx_rectangle_t rect = { 0, 0, 0, 0 };

                    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                    vx_map_id map_id;

                    void* plane_ptr = 0;

                    vx_uint8* ptr = (vx_uint8*)prev_ptrs[n];

                    EXPECT_EQ_PTR(mem2_ptrs[n], ptr);

                    ct_free_mem(ptr);
                    prev_ptrs[n] = NULL;
                    mem2_ptrs[n] = NULL;

                    VX_CALL(vxGetValidRegionImage(roi2, &rect));

                    EXPECT_EQ_INT(VX_ERROR_NO_MEMORY, vxMapImagePatch(roi2, &rect, n, &map_id, &addr, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
                }
            }

            VX_CALL(vxReleaseImage(&roi1));
            VX_CALL(vxReleaseImage(&roi2));
            ASSERT(roi1 == 0);
            ASSERT(roi2 == 0);
        }
    }
    else
    {
        /* replace image ptrs */
        VX_CALL(vxSwapImageHandle(image1, mem2_ptrs, prev_ptrs, nplanes1));
        ASSERT_NO_FAILURE(tst2 = ct_image_from_vx_image(image1));

        /* 1. verify content of image is changed to the second image */
        EXPECT_EQ_CTIMAGE(tst2, img2);

        /* 2. verify we get back original image ptrs */
        for (n = 0; n < nplanes1; n++)
        {
            EXPECT_EQ_PTR(mem1_ptrs[n], prev_ptrs[n]);
        }

        /* 3. verify we get back content of the original image */
        {
            vx_image tmp = 0;
            EXPECT_VX_OBJECT(tmp = vxCreateImageFromHandle(context, arg_->format, addr1, prev_ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);
            ASSERT_NO_FAILURE(tst1 = ct_image_from_vx_image(tmp));
            EXPECT_EQ_CTIMAGE(tst1, img1);
            VX_CALL(vxSwapImageHandle(tmp, NULL, NULL, nplanes1));
            VX_CALL(vxReleaseImage(&tmp));
        }

        /* 4. check if image ptrs were replaced */
        for (n = 0; n < nplanes1; n++)
        {
            if (prev_ptrs[n] != NULL)
            {
                vx_rectangle_t rect = { 0, 0, 0, 0 };

                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                vx_map_id map_id;

                void* plane_ptr = 0;

                VX_CALL(vxGetValidRegionImage(image1, &rect));

                VX_CALL(vxMapImagePatch(image1, &rect, n, &map_id, &addr, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

                EXPECT_EQ_PTR(mem2_ptrs[n], plane_ptr);

                VX_CALL(vxUnmapImagePatch(image1, map_id));
            }
        }

        /* reclaim image ptrs */
        VX_CALL(vxSwapImageHandle(image1, NULL, prev_ptrs, nplanes1));
        VX_CALL(vxSwapImageHandle(image2, NULL, NULL, nplanes2));

        /* 5. check if image ptrs were reclaimed */
        for (n = 0; n < nplanes1; n++)
        {
            if (prev_ptrs[n] != NULL)
            {
                vx_rectangle_t rect = { 0, 0, 0, 0 };

                vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;

                vx_map_id map_id;

                void* plane_ptr = 0;

                vx_uint8* ptr = (vx_uint8*)prev_ptrs[n];

                EXPECT_EQ_PTR(mem2_ptrs[n], ptr);

                ct_free_mem(ptr);
                prev_ptrs[n] = NULL;
                mem2_ptrs[n] = NULL;

                VX_CALL(vxGetValidRegionImage(image1, &rect));

                EXPECT_EQ_INT(VX_ERROR_NO_MEMORY, vxMapImagePatch(image1, &rect, n, &map_id, &addr, &plane_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
            }
        }
    }

    for (n = 0; n < VX_PLANE_MAX; n++)
    {
        if (mem1_ptrs[n] != NULL)
        {
            ct_free_mem(mem1_ptrs[n]);
            mem1_ptrs[n] = NULL;
        }
    }

    VX_CALL(vxReleaseImage(&image1));
    VX_CALL(vxReleaseImage(&image2));

    ASSERT(image1 == 0);
    ASSERT(image2 == 0);
} /* testSwapImageHandle() */

TEST_WITH_ARG(Image, testFormatImagePatchAddress1d, ImageGenerator_Arg,
    TEST_IMAGE_RANDOM_IMAGE_PARAMETERS
)
{
    vx_uint8* p1;
    vx_uint8* p2;
    vx_uint32 i;
    vx_int32  j;
    vx_uint32 n;
    vx_uint32 nplanes;
    vx_context context = context_->vx_context_;
    vx_image image1 = 0;
    vx_image image2 = 0;
    vx_rectangle_t rect = { 0, 0, 0, 0 };
    vx_imagepatch_addressing_t addr1[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    vx_imagepatch_addressing_t addr2[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };

    vx_map_id map_id1;
    vx_map_id map_id2;

    CT_Image src = NULL;
    CT_Image tst = NULL;

    void* ptrs1[VX_PLANE_MAX] = { 0, 0, 0, 0 };
    void* ptrs2[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    ASSERT_NO_FAILURE(src = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    image1 = ct_image_to_vx_image(src, context);
    ASSERT_VX_OBJECT(image1, VX_TYPE_IMAGE);

    image2 = vxCreateImage(context, arg_->width, arg_->height, arg_->format);
    ASSERT_VX_OBJECT(image2, VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(nplanes = ct_get_num_planes(arg_->format));

    for (n = 0; n < nplanes; n++)
    {
        rect.start_x = 0;
        rect.start_y = 0;
        rect.end_x = src->width;
        rect.end_y = src->height;

        VX_CALL(vxMapImagePatch(image1, &rect, n, &map_id1, &addr1[n], &ptrs1[n], VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        VX_CALL(vxMapImagePatch(image2, &rect, n, &map_id2, &addr2[n], &ptrs2[n], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0));

        /* use linear addressing function */
        for (i = 0; i < addr1[n].dim_x*addr1[n].dim_y; i += addr1[n].step_x)
        {
            p1 = (vx_uint8*)vxFormatImagePatchAddress1d(ptrs1[n], i, &addr1[n]);
            p2 = (vx_uint8*)vxFormatImagePatchAddress1d(ptrs2[n], i, &addr2[n]);
            for (j = 0; j < addr1[n].stride_x; j++)
                p2[j] = p1[j];
            if (addr1[n].stride_x == 0 && addr1[n].stride_x_bits == 1)  // VX_DF_IMAGE_U1 image
            {
                vx_uint8 x = i % addr1[n].dim_x;
                p2[0] = (p2[0] & ~(1 << (x % 8))) |
                        (p1[0] &  (1 << (x % 8)));
            }
        }

        VX_CALL(vxUnmapImagePatch(image1, map_id1));
        VX_CALL(vxUnmapImagePatch(image2, map_id2));
    }

    ASSERT_NO_FAILURE(tst = ct_image_from_vx_image(image2));

    ASSERT_EQ_CTIMAGE(tst, src);

    VX_CALL(vxReleaseImage(&image1));
    VX_CALL(vxReleaseImage(&image2));

    ASSERT(image1 == 0);
    ASSERT(image2 == 0);
} /* testFormatImagePatchAddress1d() */

TEST_WITH_ARG(Image, testConvert_CT_Image, ImageFormat_Arg,
    IMAGE_FORMAT_PARAMETERS_BASELINE,
    ARG("_U1_/VX_DF_IMAGE_U1", VX_DF_IMAGE_U1),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0,
               image2  = 0;
    CT_Image   ctimg   = 0,
               ctimg2  = 0;

    image = vxCreateImage(context, 16, 16, arg_->format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(ct_fill_image_random(image, &CT()->seed_));

    ASSERT_NO_FAILURE(ctimg = ct_image_from_vx_image(image));

    ASSERT_NO_FAILURE(image2 = ct_image_to_vx_image(ctimg, context));

    ASSERT_NO_FAILURE(ctimg2 = ct_image_from_vx_image(image2));

    ASSERT_EQ_CTIMAGE(ctimg, ctimg2);

    VX_CALL(vxReleaseImage(&image));
    VX_CALL(vxReleaseImage(&image2));

    ASSERT(image == 0);
    ASSERT(image2 == 0);
} /* testConvert_CT_Image() */

TEST_WITH_ARG(Image, testvxSetImagePixelValues, ImageFormat_Arg,
    IMAGE_FORMAT_PARAMETERS_BASELINE,
    ARG("_U1_/VX_DF_IMAGE_U1", VX_DF_IMAGE_U1),
)
{
    vx_context  context = context_->vx_context_;
    vx_image    image   = 0;
    CT_Image   ctimg   = 0;
    CT_Image   refimg  = 0;
    int i;
    vx_df_image format  = arg_->format;

    image = vxCreateImage(context, 640, 480, format);
    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    vx_pixel_value_t vals;

    vals.reserved[0] = 0x11;
    vals.reserved[1] = 0x22;
    vals.reserved[2] = 0x33;
    vals.reserved[3] = 0x44;

    vx_status status = vxSetImagePixelValues(image, &vals);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);

    ASSERT_NO_FAILURE(ctimg = ct_image_from_vx_image(image));

    ASSERT_NO_FAILURE(refimg = ct_allocate_image(640, 480, arg_->format));

    switch (arg_->format)
    {
        case VX_DF_IMAGE_U1:
            ct_memset(refimg->data.y, (vals.U1 ? 0xFF : 0x00), ((640 + 7) / 8) * 480);    // Set 8 pixels at a time
            break;
        case VX_DF_IMAGE_U8:
            ct_memset(refimg->data.y, vals.U8, 640*480);
            break;
        case VX_DF_IMAGE_U16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u16[i] = vals.U16;
            break;
        case VX_DF_IMAGE_S16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s16[i] = vals.S16;
            break;
        case VX_DF_IMAGE_U32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u32[i] = vals.U32;
            break;
        case VX_DF_IMAGE_S32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s32[i] = vals.S32;
            break;
        case VX_DF_IMAGE_RGB:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgb[i].r = vals.RGB[0];
                refimg->data.rgb[i].g = vals.RGB[1];
                refimg->data.rgb[i].b = vals.RGB[2];
            }
            break;
        case VX_DF_IMAGE_RGBX:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgbx[i].r = vals.RGBX[0];
                refimg->data.rgbx[i].g = vals.RGBX[1];
                refimg->data.rgbx[i].b = vals.RGBX[2];
                refimg->data.rgbx[i].x = vals.RGBX[3];
            }
            break;
        case VX_DF_IMAGE_YUV4:
            ct_memset(refimg->data.y + 640*480*0, vals.YUV[0], 640*480);
            ct_memset(refimg->data.y + 640*480*1, vals.YUV[1], 640*480);
            ct_memset(refimg->data.y + 640*480*2, vals.YUV[2], 640*480);
            break;
        case VX_DF_IMAGE_IYUV:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            ct_memset(refimg->data.y + 640*480, vals.YUV[1], 640/2*480/2);
            ct_memset(refimg->data.y + 640*480 + 640/2*480/2, vals.YUV[2], 640/2*480/2);
            break;
        case VX_DF_IMAGE_NV12:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.YUV[1];
                refimg->data.y[640*480 + 2 * i + 1] = vals.YUV[2];
            }
            break;
        case VX_DF_IMAGE_NV21:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.YUV[2];
                refimg->data.y[640*480 + 2 * i + 1] = vals.YUV[1];
            }
            break;
        case VX_DF_IMAGE_YUYV:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.yuyv[i].y0 = vals.YUV[0];
                refimg->data.yuyv[i].y1 = vals.YUV[0];
                refimg->data.yuyv[i].u = vals.YUV[1];
                refimg->data.yuyv[i].v = vals.YUV[2];
            }
            break;
        case VX_DF_IMAGE_UYVY:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.uyvy[i].y0 = vals.YUV[0];
                refimg->data.uyvy[i].y1 = vals.YUV[0];
                refimg->data.uyvy[i].u = vals.YUV[1];
                refimg->data.uyvy[i].v = vals.YUV[2];
            }
            break;
    };

    EXPECT_EQ_CTIMAGE(refimg, ctimg);

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);
} /* testvxSetImagePixelValues() */

TEST_WITH_ARG(Image, testUniformImage, ImageFormat_Arg,
    IMAGE_FORMAT_PARAMETERS_BASELINE,
    ARG("_U1_/VX_DF_IMAGE_U1", VX_DF_IMAGE_U1),
)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    CT_Image   ctimg   = 0;
    CT_Image   refimg  = 0;
    int i;

    vx_pixel_value_t vals;

    vals.reserved[0] = 0x11;
    vals.reserved[1] = 0x22;
    vals.reserved[2] = 0x33;
    vals.reserved[3] = 0x44;

    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, 640, 480, arg_->format, &vals), VX_TYPE_IMAGE);
    ASSERT_NO_FAILURE(ctimg = ct_image_from_vx_image(image));

    ASSERT_NO_FAILURE(refimg = ct_allocate_image(640, 480, arg_->format));

    switch (arg_->format)
    {
        case VX_DF_IMAGE_U1:
            ct_memset(refimg->data.y, (vals.U1 ? 0xFF : 0x00), ((640 + 7) / 8) * 480);    // Set 8 pixels at a time
            break;
        case VX_DF_IMAGE_U8:
            ct_memset(refimg->data.y, vals.U8, 640*480);
            break;
        case VX_DF_IMAGE_U16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u16[i] = vals.U16;
            break;
        case VX_DF_IMAGE_S16:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s16[i] = vals.S16;
            break;
        case VX_DF_IMAGE_U32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.u32[i] = vals.U32;
            break;
        case VX_DF_IMAGE_S32:
            for (i = 0; i < 640*480; ++i)
                refimg->data.s32[i] = vals.S32;
            break;
        case VX_DF_IMAGE_RGB:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgb[i].r = vals.RGB[0];
                refimg->data.rgb[i].g = vals.RGB[1];
                refimg->data.rgb[i].b = vals.RGB[2];
            }
            break;
        case VX_DF_IMAGE_RGBX:
            for (i = 0; i < 640*480; ++i)
            {
                refimg->data.rgbx[i].r = vals.RGBX[0];
                refimg->data.rgbx[i].g = vals.RGBX[1];
                refimg->data.rgbx[i].b = vals.RGBX[2];
                refimg->data.rgbx[i].x = vals.RGBX[3];
            }
            break;
        case VX_DF_IMAGE_YUV4:
            ct_memset(refimg->data.y + 640*480*0, vals.YUV[0], 640*480);
            ct_memset(refimg->data.y + 640*480*1, vals.YUV[1], 640*480);
            ct_memset(refimg->data.y + 640*480*2, vals.YUV[2], 640*480);
            break;
        case VX_DF_IMAGE_IYUV:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            ct_memset(refimg->data.y + 640*480, vals.YUV[1], 640/2*480/2);
            ct_memset(refimg->data.y + 640*480 + 640/2*480/2, vals.YUV[2], 640/2*480/2);
            break;
        case VX_DF_IMAGE_NV12:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.YUV[1];
                refimg->data.y[640*480 + 2 * i + 1] = vals.YUV[2];
            }
            break;
        case VX_DF_IMAGE_NV21:
            ct_memset(refimg->data.y, vals.YUV[0], 640 * 480);
            for (i = 0; i < 640/2*480/2; ++i)
            {
                refimg->data.y[640*480 + 2 * i + 0] = vals.YUV[2];
                refimg->data.y[640*480 + 2 * i + 1] = vals.YUV[1];
            }
            break;
        case VX_DF_IMAGE_YUYV:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.yuyv[i].y0 = vals.YUV[0];
                refimg->data.yuyv[i].y1 = vals.YUV[0];
                refimg->data.yuyv[i].u = vals.YUV[1];
                refimg->data.yuyv[i].v = vals.YUV[2];
            }
            break;
        case VX_DF_IMAGE_UYVY:
            for (i = 0; i < 640/2*480; ++i)
            {
                refimg->data.uyvy[i].y0 = vals.YUV[0];
                refimg->data.uyvy[i].y1 = vals.YUV[0];
                refimg->data.uyvy[i].u = vals.YUV[1];
                refimg->data.uyvy[i].v = vals.YUV[2];
            }
            break;
    };

    EXPECT_EQ_CTIMAGE(refimg, ctimg);

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);
} /* testUniformImage() */

#define IMAGE_SIZE_X 320
#define IMAGE_SIZE_Y 200
#define PATCH_SIZE_X 33
#define PATCH_SIZE_Y 12
#define PATCH_ORIGIN_X 51
#define PATCH_ORIGIN_Y 15

TEST(Image, testAccessCopyWrite)
{
    vx_context context = context_->vx_context_;
    vx_uint8 *localPatchDense  = (vx_uint8*)ct_alloc_mem(PATCH_SIZE_X*PATCH_SIZE_Y*sizeof(vx_uint8));
    vx_uint8 *localPatchSparse = (vx_uint8*)ct_alloc_mem(PATCH_SIZE_X*PATCH_SIZE_Y*3*3*sizeof(vx_uint8));
    vx_image image;
    int x, y;
    vx_map_id map_id;

    ASSERT_VX_OBJECT( image = vxCreateImage(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    /* Image Initialization */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxMapImagePatch(image, &rectFull, 0, &map_id, &addrFull, (void **)&p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                *pPixel = 0;
            }
        }
        VX_CALL( vxUnmapImagePatch(image, map_id));

        /* Buffer Initialization */
        for (y = 0; y < PATCH_SIZE_Y; y++) {
            for (x = 0; x < PATCH_SIZE_X; x++) {
                localPatchDense[x + y*PATCH_SIZE_X] = x + y;

                localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] = 2*(x + y);
                localPatchSparse[(3*x+1) + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+1) + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+1) + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[(3*x+2) + (3*y+2)*3*PATCH_SIZE_X] = 0;
            }
        }
    }

    /* Write, COPY, No spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                sizeof(vx_uint8), PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchDense[0];
        VX_CALL( vxCopyImagePatch(image, &rectPatch, 0, &addrPatch, (void *)p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT(p == &localPatchDense[0]);
    }
    /* Check (MAP) */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxMapImagePatch(image, &rectFull, 0, &map_id, &addrFull, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                if ( (x<PATCH_ORIGIN_X) || (x>=PATCH_ORIGIN_X+PATCH_SIZE_X) ||
                     (y<PATCH_ORIGIN_Y) || (y>=PATCH_ORIGIN_Y+PATCH_SIZE_Y) ) {
                    ASSERT( *pPixel == 0);
                }
                else {
                    ASSERT( *pPixel == (x + y - PATCH_ORIGIN_X - PATCH_ORIGIN_Y));
                }
            }
        }
        VX_CALL( vxUnmapImagePatch(image, map_id));
    }


    /* Write, COPY, Spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                3*sizeof(vx_uint8), 3*3*PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchSparse[0];
        VX_CALL( vxCopyImagePatch(image, &rectPatch, 0, &addrPatch, (void *)p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT(p == &localPatchSparse[0]);
    }
    /* Check (MAP) */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxMapImagePatch(image, &rectFull, 0, &map_id, &addrFull, (void **)&p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                if ( (x<PATCH_ORIGIN_X) || (x>=PATCH_ORIGIN_X+PATCH_SIZE_X) ||
                     (y<PATCH_ORIGIN_Y) || (y>=PATCH_ORIGIN_Y+PATCH_SIZE_Y) ) {
                    ASSERT( *pPixel == 0);
                }
                else {
                    ASSERT( *pPixel == (2*(x + y - PATCH_ORIGIN_X - PATCH_ORIGIN_Y)));
                }
            }
        }
        VX_CALL( vxUnmapImagePatch(image, map_id));
    }



    VX_CALL( vxReleaseImage(&image) );
    ASSERT( image == 0);

    ct_free_mem(localPatchDense);
    ct_free_mem(localPatchSparse);
} /* testAccessCopyWrite() */

TEST(Image, testAccessCopyRead)
{
    vx_context context = context_->vx_context_;
    vx_uint8 *localPatchDense  = (vx_uint8*)ct_alloc_mem(PATCH_SIZE_X*PATCH_SIZE_Y*sizeof(vx_uint8));
    vx_uint8 *localPatchSparse = (vx_uint8*)ct_alloc_mem(PATCH_SIZE_X*PATCH_SIZE_Y*3*3*sizeof(vx_uint8));
    vx_image image;
    int x, y;
    vx_map_id map_id;

    ASSERT_VX_OBJECT( image = vxCreateImage(context, IMAGE_SIZE_X, IMAGE_SIZE_Y, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    /* Image Initialization */
    {
        vx_rectangle_t rectFull = {0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y};
        vx_imagepatch_addressing_t addrFull;
        vx_uint8 *p = NULL, *pLine, *pPixel = NULL;
        VX_CALL( vxMapImagePatch(image, &rectFull, 0, &map_id, &addrFull, (void **)&p, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        for (y = 0, pLine = p; y < IMAGE_SIZE_Y; y++, pLine += addrFull.stride_y) {
            for (x = 0, pPixel = pLine; x < IMAGE_SIZE_X; x++, pPixel += addrFull.stride_x) {
                *pPixel = x + y;
            }
        }
        VX_CALL( vxUnmapImagePatch(image, map_id));

        /* Buffer Initialization */
        for (y = 0; y < PATCH_SIZE_Y; y++) {
            for (x = 0; x < PATCH_SIZE_X; x++) {
                localPatchDense[x + y*PATCH_SIZE_X] = 0;

                localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + 3*y*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + (3*y+1)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+1 + (3*y+2)*3*PATCH_SIZE_X] = 0;
                localPatchSparse[3*x+2 + (3*y+2)*3*PATCH_SIZE_X] = 0;
            }
        }
    }

    /* READ, COPY, No spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                sizeof(vx_uint8), PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchDense[0];
        VX_CALL( vxCopyImagePatch(image, &rectPatch, 0, &addrPatch, (void *)p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT(p == &localPatchDense[0]);
        ASSERT(addrPatch.stride_x == sizeof(vx_uint8));
        ASSERT(addrPatch.stride_y == PATCH_SIZE_X*sizeof(vx_uint8));
    }
    /* Check */
    for (y = 0; y < PATCH_SIZE_Y; y++) {
        for (x = 0; x < PATCH_SIZE_X; x++) {
            ASSERT(localPatchDense[x + y*PATCH_SIZE_X] == x + y + PATCH_ORIGIN_X + PATCH_ORIGIN_Y);
        }
    }

    /* READ, COPY, Spacing */
    {
        vx_rectangle_t rectPatch = {PATCH_ORIGIN_X, PATCH_ORIGIN_Y, PATCH_ORIGIN_X+PATCH_SIZE_X, PATCH_ORIGIN_Y+PATCH_SIZE_Y};
        vx_imagepatch_addressing_t addrPatch = {PATCH_SIZE_X, PATCH_SIZE_Y,
                                                3*sizeof(vx_uint8), 3*3*PATCH_SIZE_X*sizeof(vx_uint8),
                                                VX_SCALE_UNITY, VX_SCALE_UNITY,
                                                1, 1 };
        vx_uint8 *p = &localPatchSparse[0];
        VX_CALL( vxCopyImagePatch(image, &rectPatch, 0, &addrPatch, (void *)p, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        ASSERT(p == &localPatchSparse[0]);
        ASSERT(addrPatch.stride_x == 3*sizeof(vx_uint8));
        ASSERT(addrPatch.stride_y == 3*3*PATCH_SIZE_X*sizeof(vx_uint8));
    }
    /* Check */
    for (y = 0; y < PATCH_SIZE_Y; y++) {
        for (x = 0; x < PATCH_SIZE_X; x++) {
            ASSERT(localPatchSparse[3*x + 3*y*3*PATCH_SIZE_X] == x + y + PATCH_ORIGIN_X + PATCH_ORIGIN_Y);
            ASSERT(localPatchSparse[3*x+1 + 3*y*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + 3*y*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+1 + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + (3*y+1)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x + (3*y+2)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+1 + (3*y+2)*3*PATCH_SIZE_X] == 0);
            ASSERT(localPatchSparse[3*x+2 + (3*y+2)*3*PATCH_SIZE_X] == 0);
        }
    }

    VX_CALL( vxReleaseImage(&image) );
    ASSERT( image == 0);

    ct_free_mem(localPatchDense);
    ct_free_mem(localPatchSparse);
} /* testAccessCopyRead() */

TEST(Image, testAccessCopyWriteUniformImage)
{
    vx_context context = context_->vx_context_;
    vx_image image = 0;
    vx_image roi_image = 0;
    vx_uint32 width = 320;
    vx_uint32 height = 240;
    vx_uint32 roi_width = 128;
    vx_uint32 roi_height = 128;
    vx_map_id map_id;

    vx_pixel_value_t vals = {{0xFF}};
    vx_rectangle_t rect = {0, 0, width, height};
    vx_rectangle_t roi_rect = {0, 0, roi_width, roi_height};
    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
    vx_imagepatch_addressing_t roi_addr = VX_IMAGEPATCH_ADDR_INIT;
    roi_addr.dim_x = roi_width;
    roi_addr.dim_y = roi_height;
    roi_addr.stride_x = 1;
    roi_addr.stride_y = roi_width;

    vx_uint8 *internal_data = NULL;
    vx_uint8 *external_data = (vx_uint8 *)ct_alloc_mem(roi_width * roi_height * sizeof(vx_uint8));

    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, width, height, VX_DF_IMAGE_U8, &vals), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(roi_image = vxCreateImageFromROI(image, &roi_rect), VX_TYPE_IMAGE);

    // Can get read-access, cannot get write-access
    vx_status status = vxMapImagePatch(image, &rect, 0, &map_id, &addr, (void **)&internal_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    status = vxUnmapImagePatch(image, map_id);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    internal_data = NULL;
    status = vxMapImagePatch(image, &rect, 0, &map_id, &addr, (void **)&internal_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    ASSERT_NE_VX_STATUS(VX_SUCCESS, status);

    // Reading from the image is allowed, writing to the image is not allowed
    status = vxCopyImagePatch(image, &roi_rect, 0, &roi_addr, (void *)external_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    status = vxCopyImagePatch(image, &roi_rect, 0, &roi_addr, (void *)external_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    ASSERT_NE_VX_STATUS(VX_SUCCESS, status);

    // Test ROI image(from uniform image), behaviour must be equal to uniform image
    // Can get read-access, cannot get write-access
    internal_data = NULL;
    status = vxMapImagePatch(roi_image, &roi_rect, 0, &map_id, &addr, (void **)&internal_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    status = vxUnmapImagePatch(roi_image, map_id);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    internal_data = NULL;
    status = vxMapImagePatch(roi_image, &roi_rect, 0, &map_id, &addr, (void **)&internal_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    ASSERT_NE_VX_STATUS(VX_SUCCESS, status);

    // Reading from the image is allowed, writing to the image is not allowed
    status = vxCopyImagePatch(roi_image, &roi_rect, 0, &roi_addr, (void *)external_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    status = vxCopyImagePatch(roi_image, &roi_rect, 0, &roi_addr, (void *)external_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    ASSERT_NE_VX_STATUS(VX_SUCCESS, status);

    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxReleaseImage(&image));
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, vxReleaseImage(&roi_image));
    ASSERT(image == 0);
    ASSERT(roi_image == 0);

    ct_free_mem(external_data);
} /* testAccessCopyWriteUniformImage() */

TEST(Image, testQueryImage)
{
    vx_context context = context_->vx_context_;
    vx_image   image   = 0;
    vx_df_image format = 0;
    vx_uint32 width = 0;
    vx_uint32 height = 0;
    vx_size planes = 0;
    vx_enum space = 0;
    vx_enum range = 0;
    vx_enum memory_type = 0;

    image = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8);

    ASSERT_VX_OBJECT(image, VX_TYPE_IMAGE);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_FORMAT,  &format,  sizeof(format)));
    ASSERT_EQ_INT(VX_DF_IMAGE_U8, format);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_WIDTH,  &width,  sizeof(width)));
    ASSERT_EQ_INT(640, width);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_HEIGHT,  &height,  sizeof(height)));
    ASSERT_EQ_INT(480, height);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_PLANES,  &planes,  sizeof(planes)));
    ASSERT_EQ_INT(1, planes);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_SPACE,  &space,  sizeof(space)));
    ASSERT_EQ_INT(VX_COLOR_SPACE_NONE, space);

    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_RANGE,  &range,  sizeof(range)));
    ASSERT_EQ_INT(VX_CHANNEL_RANGE_FULL, range);
/* commented out untill spec clarifies VX_IMAGE_SIZE in more details:
   - is that memory size actually allocated
   - is that min number of bytes to store pixels
   - is that number of bytes it would occupy in memory once allocated (with strides)
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_SIZE,  &image_size,  sizeof(image_size)));
    ASSERT_EQ_INT(0, image_size);
*/
    ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxQueryImage(image, VX_IMAGE_MEMORY_TYPE,  &memory_type,  sizeof(memory_type)));
    ASSERT_EQ_INT(VX_MEMORY_TYPE_NONE, memory_type);

    VX_CALL(vxReleaseImage(&image));
} /* testQueryImage() */


/* ***************************************************************************
//  local auxiliary functions
*/

/*
// Generate input random pixel values
*/
static CT_Image own_generate_rand_image(const char* fileName, int width, int height, vx_df_image format)
{
    CT_Image image;

    if (format == VX_DF_IMAGE_U1)
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 2));
    else
        ASSERT_NO_FAILURE_(return 0, image = ct_allocate_ct_image_random(width, height, format, &CT()->seed_, 0, 256));

    return image;
} /* own_generate_rand_image() */

static vx_uint32 own_plane_subsampling_x(vx_df_image format, vx_uint32 plane)
{
    int subsampling_x = 0;

    switch (format)
    {
    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        subsampling_x = (0 == plane) ? 1 : 2;
        break;

    default:
        subsampling_x = 1;
        break;
    }

    return subsampling_x;
} /* own_plane_subsampling_x() */

static vx_uint32 own_plane_subsampling_y(vx_df_image format, vx_uint32 plane)
{
    int subsampling_y = 0;

    switch (format)
    {
    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        subsampling_y = (0 == plane) ? 1 : 2;
        break;

    default:
        subsampling_y = 1;
        break;
    }

    return subsampling_y;
} /* own_plane_subsampling_y() */

static vx_uint32 own_elem_size(vx_df_image format, vx_uint32 plane)
{
    int channel_step_x = 0;

    switch (format)
    {
    case VX_DF_IMAGE_U1:
        channel_step_x = 0;
        break;

    case VX_DF_IMAGE_U8:
        channel_step_x = 1;
        break;

    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
        channel_step_x = 2;
        break;

    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGBX:
        channel_step_x = 4;
        break;

    case VX_DF_IMAGE_RGB:
        channel_step_x = 3;
        break;

    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        channel_step_x = 2;
        break;

    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_YUV4:
        channel_step_x = 1;
        break;

    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        channel_step_x = (0 == plane) ? 1 : 2;
        break;

    default:
        channel_step_x = 0;
    }

    return channel_step_x;
} /* own_elem_size() */

static uint32_t own_stride_bytes(vx_df_image format, int step)
{
    uint32_t factor = 0;

    switch (format)
    {
    case VX_DF_IMAGE_U1:
        return (step + 7) / 8;

    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV:
        factor = 1;
        break;

    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        factor = 2;
        break;

    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGBX:
        factor = 4;
        break;

    case VX_DF_IMAGE_RGB:
        factor = 3;
        break;

    default:
        ASSERT_(return 0, 0);
    }

    return step*factor;
} /* own_stride_bytes() */

static int own_get_channel_step_x(vx_df_image format, vx_enum channel)
{
    switch (format)
    {
    case VX_DF_IMAGE_U1:
        return 0;

    case VX_DF_IMAGE_U8:
        return 1;

    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
        return 2;

    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGBX:
        return 4;

    case VX_DF_IMAGE_RGB:
        return 3;

    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        if (channel == VX_CHANNEL_Y)
            return 2;
        return 4;

    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_YUV4:
        return 1;

    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        if (channel == VX_CHANNEL_Y)
            return 1;
        return 2;

    default:
        ASSERT_(return 0, 0);
    }

    return 0;
} /* own_get_channel_step_x() */

static int own_get_channel_step_y(vx_df_image format, vx_enum channel, int step)
{
    switch (format)
    {
    case VX_DF_IMAGE_U1:
        return (step + 7) / 8;

    case VX_DF_IMAGE_U8:
        return step;

    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
        return step * 2;

    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGBX:
        return step * 4;

    case VX_DF_IMAGE_RGB:
        return step * 3;

    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        return step * 2;

    case VX_DF_IMAGE_IYUV:
        return (channel == VX_CHANNEL_Y) ? step : step / 2;

    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        return step;

    default:
        ASSERT_(return 0, 0);
    }

    return 0;
} /* own_get_channel_step_y() */

static int own_get_channel_subsampling_x(vx_df_image format, vx_enum channel)
{
    if (channel == VX_CHANNEL_Y)
        return 1;

    switch (format)
    {
    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        return 2;
    }

    return 1;
} /* own_get_channel_subsampling_x() */

int own_get_channel_subsampling_y(vx_df_image format, vx_enum channel)
{
    if (channel == VX_CHANNEL_Y)
        return 1;

    switch (format)
    {
    case VX_DF_IMAGE_IYUV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        return 2;

    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_UYVY:
        return 1;
    }

    return 1;
} /* own_get_channel_subsampling_y() */

static unsigned int own_image_bits_per_pixel(vx_df_image format, unsigned int p)
{
    switch (format)
    {
    case VX_DF_IMAGE_U1:
        return 1 * 1;

    case VX_DF_IMAGE_U8:
        return 8 * 1;

    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_UYVY:
    case VX_DF_IMAGE_YUYV:
        return 8 * 2;

    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
    case VX_DF_IMAGE_RGBX:
        return 8 * 4;

    case VX_DF_IMAGE_RGB:
    case VX_DF_IMAGE_YUV4:
        return 8 * 3;

    case VX_DF_IMAGE_IYUV:
        return 8 * 3 / 2;

    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
        if (p == 0)
            return 8 * 1;
        else
            return 8 * 2;

    default:
        CT_RecordFailure();
        return 0;
    };
} /* own_image_bits_per_pixel() */

static size_t own_plane_size(uint32_t width, uint32_t height, unsigned int p, vx_df_image format)
{
    if (format == VX_DF_IMAGE_U1)
    {
        /* round rows up to full bytes */
        size_t rowSize = (size_t)(width * own_image_bits_per_pixel(format, p) + 7) / 8;
        return (size_t)(rowSize * height);
    }
    else
    {
        return (size_t)(width * height * own_image_bits_per_pixel(format, p) / 8);
    }
} /* own_plane_size() */

/*
// Allocates image plane pointers from user controlled memory according to format, width, height params
// and initialize with some value
*/
static void own_allocate_image_ptrs(
    vx_df_image format, int width, int height,
    vx_uint32* nplanes, void* ptrs[], vx_imagepatch_addressing_t addr[],
    vx_pixel_value_t* val)
{
    unsigned int p;
    int channel[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    switch (format)
    {
    case VX_DF_IMAGE_U1:
    case VX_DF_IMAGE_U8:
    case VX_DF_IMAGE_U16:
    case VX_DF_IMAGE_S16:
    case VX_DF_IMAGE_U32:
    case VX_DF_IMAGE_S32:
        channel[0] = VX_CHANNEL_0;
        break;

    case VX_DF_IMAGE_RGB:
    case VX_DF_IMAGE_RGBX:
        channel[0] = VX_CHANNEL_R;
        channel[1] = VX_CHANNEL_G;
        channel[2] = VX_CHANNEL_B;
        channel[3] = VX_CHANNEL_A;
        break;

    case VX_DF_IMAGE_UYVY:
    case VX_DF_IMAGE_YUYV:
    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    case VX_DF_IMAGE_YUV4:
    case VX_DF_IMAGE_IYUV:
        channel[0] = VX_CHANNEL_Y;
        channel[1] = VX_CHANNEL_U;
        channel[2] = VX_CHANNEL_V;
        break;

    default:
        ASSERT(0);
    }

    ASSERT_NO_FAILURE(*nplanes = ct_get_num_planes(format));

    for (p = 0; p < *nplanes; p++)
    {
        size_t plane_size = 0;

        vx_uint32 subsampling_x = own_get_channel_subsampling_x(format, channel[p]);
        vx_uint32 subsampling_y = own_get_channel_subsampling_y(format, channel[p]);

        addr[p].dim_x    = width  / subsampling_x;
        addr[p].dim_y    = height / subsampling_y;
        addr[p].stride_x = own_get_channel_step_x(format, channel[p]);
        addr[p].stride_y = own_get_channel_step_y(format, channel[p], width);
        if (format == VX_DF_IMAGE_U1)
            addr[p].stride_x_bits = 1;

        plane_size = addr[p].stride_y * addr[p].dim_y;

        if (plane_size != 0)
        {
            ptrs[p] = ct_alloc_mem(plane_size);
            /* init memory */
            ct_memset(ptrs[p], val->reserved[p], plane_size);
        }
    }

    return;
} /* own_allocate_image_ptrs() */

static void mem_free(void**ptr)
{
    ct_free_mem(*ptr);
    *ptr = 0;
} /* mem_free() */

/*
// Check image patch data in user memory against constant pixel value
// Note:
//  can't use vxFormatImagePatchAddress2d for user memory layout
*/
void own_check_image_patch_uniform(vx_pixel_value_t* ref_val, void* ptr, vx_imagepatch_addressing_t* addr, vx_uint32 plane, vx_df_image format)
{
    vx_uint32 x;
    vx_uint32 y;

    for (y = 0; y < addr->dim_y; y++)
    {
        for (x = 0; x < addr->dim_x; x++)
        {
            switch (format)
            {
            case VX_DF_IMAGE_U1:
            {
                vx_uint8 offset = x % 8;
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + (x * addr->stride_x_bits) / 8);
                vx_uint8  ref = ref_val->U1 ? 1 : 0;
                ASSERT_EQ_INT(ref, (tst[0] & (1 << offset)) >> offset );
            }
            break;

            case VX_DF_IMAGE_U8:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                vx_uint8  ref = ref_val->U8;
                ASSERT_EQ_INT(ref, tst[0]);
            }
            break;

            case VX_DF_IMAGE_U16:
            {
                vx_uint16* tst = (vx_uint16*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                vx_uint16  ref = ref_val->U16;
                ASSERT_EQ_INT(ref, tst[0]);
            }
            break;

            case VX_DF_IMAGE_S16:
            {
                vx_int16* tst = (vx_int16*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                vx_int16  ref = ref_val->S16;
                ASSERT_EQ_INT(ref, tst[0]);
            }
            break;

            case VX_DF_IMAGE_U32:
            {
                vx_uint32* tst = (vx_uint32*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                vx_uint32  ref = ref_val->U32;
                ASSERT_EQ_INT(ref, tst[0]);
            }
            break;

            case VX_DF_IMAGE_S32:
            {
                vx_int32* tst = (vx_int32*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                vx_int32  ref = ref_val->S32;
                ASSERT_EQ_INT(ref, tst[0]);
            }
            break;

            case VX_DF_IMAGE_RGB:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                ASSERT_EQ_INT(ref_val->RGB[0], tst[0]);
                ASSERT_EQ_INT(ref_val->RGB[1], tst[1]);
                ASSERT_EQ_INT(ref_val->RGB[2], tst[2]);
            }
            break;

            case VX_DF_IMAGE_RGBX:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                ASSERT_EQ_INT(ref_val->RGBX[0], tst[0]);
                ASSERT_EQ_INT(ref_val->RGBX[1], tst[1]);
                ASSERT_EQ_INT(ref_val->RGBX[2], tst[2]);
                ASSERT_EQ_INT(ref_val->RGBX[3], tst[3]);
            }
            break;

            case VX_DF_IMAGE_YUYV:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);

                vx_uint8 tst_u_or_v = tst[1];

                ASSERT_EQ_INT(ref_val->YUV[0], tst[0]); // Y

                if (x & 1)
                    ASSERT_EQ_INT(ref_val->YUV[2], tst_u_or_v); // V
                else
                    ASSERT_EQ_INT(ref_val->YUV[1], tst_u_or_v); // U
            }
            break;

            case VX_DF_IMAGE_UYVY:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);

                vx_uint8 tst_u_or_v = tst[0];

                ASSERT_EQ_INT(ref_val->YUV[0], tst[1]); // Y

                if (x & 1)
                    ASSERT_EQ_INT(ref_val->YUV[2], tst_u_or_v); // V
                else
                    ASSERT_EQ_INT(ref_val->YUV[1], tst_u_or_v); // U
            }
            break;

            case VX_DF_IMAGE_NV12:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);

                if (0 == plane)
                    ASSERT_EQ_INT(ref_val->YUV[0], tst[0]); // Y
                else
                {
                    ASSERT_EQ_INT(ref_val->YUV[1], tst[0]); // U
                    ASSERT_EQ_INT(ref_val->YUV[2], tst[1]); // V
                }
            }
            break;

            case VX_DF_IMAGE_NV21:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);

                if (0 == plane)
                    ASSERT_EQ_INT(ref_val->YUV[0], tst[0]); // Y
                else
                {
                    ASSERT_EQ_INT(ref_val->YUV[1], tst[1]); // U
                    ASSERT_EQ_INT(ref_val->YUV[2], tst[0]); // V
                }
            }
            break;

            case VX_DF_IMAGE_YUV4:
            case VX_DF_IMAGE_IYUV:
            {
                vx_uint8* tst = (vx_uint8*)((vx_uint8*)ptr + y * addr->stride_y + x * addr->stride_x);
                ASSERT_EQ_INT(ref_val->YUV[plane], tst[0]);
            }
            break;

            default:
                FAIL("unexpected image format: (%.4s)", format);
            break;
            } /* switch format */
        } /* for addr.dim_x */
    } /* for addr.dim_y */

    return;
} /* own_check_image_patch_uniform() */

/*
//  Fill image patch info according to user memory layout
*/
void own_image_patch_from_ct_image(CT_Image ref, vx_imagepatch_addressing_t* ref_addr, void** ref_ptrs, vx_df_image format)
{
    switch (format)
    {
    case VX_DF_IMAGE_U1:
    {
        ref_addr[0].dim_x   = ref->width + ref->roi.x % 8;
        ref_addr[0].dim_y   = ref->height;
        ref_addr[0].stride_x = 0;
        ref_addr[0].stride_y = (ref->stride + 7) / 8;
        ref_addr[0].stride_x_bits = 1;

        ref_ptrs[0] = ref->data.y;
    }
    break;

    case VX_DF_IMAGE_U8:
    {
        ref_addr[0].dim_x   = ref->width;
        ref_addr[0].dim_y   = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.y;
    }
    break;

    case VX_DF_IMAGE_U16:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint16);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint16);

        ref_ptrs[0] = ref->data.u16;
    }
    break;

    case VX_DF_IMAGE_S16:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_int16);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_int16);

        ref_ptrs[0] = ref->data.s16;
    }
    break;

    case VX_DF_IMAGE_U32:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint32);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint32);

        ref_ptrs[0] = ref->data.u32;
    }
    break;

    case VX_DF_IMAGE_S32:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_int32);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_int32);

        ref_ptrs[0] = ref->data.s32;
    }
    break;

    case VX_DF_IMAGE_RGB:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = 3 * sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * 3 * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.rgb;
    }
    break;

    case VX_DF_IMAGE_RGBX:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = 4 * sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * 4 * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.rgbx;
    }
    break;

    case VX_DF_IMAGE_UYVY:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = 2 * sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * 2 * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.uyvy;
    }
    break;

    case VX_DF_IMAGE_YUYV:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = 2 * sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * 2 * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.yuyv;
    }
    break;

    case VX_DF_IMAGE_NV12:
    case VX_DF_IMAGE_NV21:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.yuv;

        ref_addr[1].dim_x    = ref->width / 2;
        ref_addr[1].dim_y    = ref->height / 2;
        ref_addr[1].stride_x = 2 * sizeof(vx_uint8);
        ref_addr[1].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[1] = (vx_uint8*)((vx_uint8*)ref->data.yuv + ref->height * ref->stride * sizeof(vx_uint8));
    }
    break;

    case VX_DF_IMAGE_YUV4:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[0] = (vx_uint8*)((vx_uint8*)ref->data.yuv + 0 * ref->height * ref->stride * sizeof(vx_uint8));

        ref_addr[1].dim_x    = ref->width;
        ref_addr[1].dim_y    = ref->height;
        ref_addr[1].stride_x = sizeof(vx_uint8);
        ref_addr[1].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[1] = (vx_uint8*)((vx_uint8*)ref->data.yuv + 1 * ref->height * ref->stride * sizeof(vx_uint8));

        ref_addr[2].dim_x    = ref->width;
        ref_addr[2].dim_y    = ref->height;
        ref_addr[2].stride_x = sizeof(vx_uint8);
        ref_addr[2].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[2] = (vx_uint8*)((vx_uint8*)ref->data.yuv + 2 * ref->height * ref->stride * sizeof(vx_uint8));
    }
    break;

    case VX_DF_IMAGE_IYUV:
    {
        ref_addr[0].dim_x    = ref->width;
        ref_addr[0].dim_y    = ref->height;
        ref_addr[0].stride_x = sizeof(vx_uint8);
        ref_addr[0].stride_y = ref->stride * sizeof(vx_uint8);

        ref_ptrs[0] = ref->data.yuv;

        ref_addr[1].dim_x    = ref->width / 2;
        ref_addr[1].dim_y    = ref->height / 2;
        ref_addr[1].stride_x = sizeof(vx_uint8);
        ref_addr[1].stride_y = (ref->width / 2) * sizeof(vx_uint8);

        ref_ptrs[1] = (vx_uint8*)((vx_uint8*)ref->data.yuv + ref->height * ref->stride * sizeof(vx_uint8));

        ref_addr[2].dim_x    = ref->width / 2;
        ref_addr[2].dim_y    = ref->height / 2;
        ref_addr[2].stride_x = sizeof(vx_uint8);
        ref_addr[2].stride_y = (ref->width / 2) * sizeof(vx_uint8);

        ref_ptrs[2] = (vx_uint8*)((vx_uint8*)ref->data.yuv +
            ref->height * ref->stride * sizeof(vx_uint8) +
            (ref->height / 2) * (ref->width / 2) * sizeof(vx_uint8));
    }
    break;

    default:
        FAIL("unexpected image format: (%.4s)", format);
        break;
    } /* switch format */

    return;
} /* own_image_patch_from_ct_image() */

void own_check_image_patch_plane_user_layout(CT_Image ref, vx_imagepatch_addressing_t* tst_addr, void* ptr, vx_uint32 plane, vx_df_image format)
{
    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 elem_size = own_elem_size(format, plane);

    uint32_t xROIOffset = (format == VX_DF_IMAGE_U1) ? ref->roi.x % 8 : 0;     // Offset needed for U1 ROI
    for (y = 0; y < tst_addr->dim_y; y++)
    {
        for (x = xROIOffset; x < tst_addr->dim_x + xROIOffset; x++)
        {
            switch (format)
            {
            case VX_DF_IMAGE_U1:
            {
                vx_uint8  offset  = x % 8;
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y +
                                                (x * tst_addr->stride_x_bits) / 8);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ct_stride_bytes(ref) +
                                                (x * ct_image_bits_per_pixel(VX_DF_IMAGE_U1)) / 8);
                ASSERT_EQ_INT((ref_ptr[0] & (1 << offset)) >> offset,
                              (tst_ptr[0] & (1 << offset)) >> offset);
            }
            break;

            case VX_DF_IMAGE_U8:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_U16:
            {
                vx_uint16* tst_ptr = (vx_uint16*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint16* ref_ptr = (vx_uint16*)((vx_uint8*)ref->data.u16 + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_S16:
            {
                vx_int16* tst_ptr = (vx_int16*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_int16* ref_ptr = (vx_int16*)((vx_uint8*)ref->data.s16 + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_U32:
            {
                vx_uint32* tst_ptr = (vx_uint32*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint32* ref_ptr = (vx_uint32*)((vx_uint8*)ref->data.u32 + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_S32:
            {
                vx_int32* tst_ptr = (vx_int32*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_int32* ref_ptr = (vx_int32*)((vx_uint8*)ref->data.s32 + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_RGB:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgb + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                ASSERT_EQ_INT(ref_ptr[2], tst_ptr[2]);
            }
            break;

            case VX_DF_IMAGE_RGBX:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgbx + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                ASSERT_EQ_INT(ref_ptr[2], tst_ptr[2]);
                ASSERT_EQ_INT(ref_ptr[3], tst_ptr[3]);
            }
            break;

            case VX_DF_IMAGE_YUYV:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.yuyv + y * ref->stride * elem_size + x * elem_size);

                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // Y
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]); // U or V
            }
            break;

            case VX_DF_IMAGE_UYVY:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.uyvy + y * ref->stride * elem_size + x * elem_size);

                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // U or V
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]); // Y
            }
            break;

            case VX_DF_IMAGE_NV12:
            case VX_DF_IMAGE_NV21:
            {
                vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride * elem_size + x * elem_size);

                if (0 == plane)
                    ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // Y
                else
                {
                    ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                    ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                }
            }
            break;

            case VX_DF_IMAGE_IYUV:
            {
                vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_YUV4:
            {
                vx_uint8* tst_ptr = (vx_uint8*)((vx_uint8*)ptr + y * tst_addr->stride_y + x * tst_addr->stride_x);
                vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * ref->stride * elem_size + x * elem_size);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            default:
                FAIL("unexpected image format: (%.4s)", format);
                break;
            } /* switch format */
        } /* for tst_addr.dim_x */
    } /* for tst_addr.dim_y */

    return;
} /* own_check_image_patch_plane_user_layout() */

void own_check_image_patch_plane_vx_layout(CT_Image ctimg, vx_imagepatch_addressing_t* vx_addr, void* p_vx_base, vx_uint32 plane, vx_df_image format)
{
    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 ct_elem_size = own_elem_size(format, plane);
    void*     p_ct_base = ct_image_get_plane_base(ctimg, plane);

    for (y = 0; y < vx_addr->dim_y; y += vx_addr->step_y)
    {
        for (x = 0; x < vx_addr->dim_x; x += vx_addr->step_x)
        {
            switch (format)
            {
            case VX_DF_IMAGE_U1:
            {
                vx_uint8  offset  = x % 8;
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ct_stride_bytes(ctimg) +
                                                (x * ct_image_bits_per_pixel(VX_DF_IMAGE_U1)) / 8);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT((ref_ptr[0] & (1 << offset)) >> offset,
                              (tst_ptr[0] & (1 << offset)) >> offset);
            }
            break;

            case VX_DF_IMAGE_U8:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_U16:
            {
                vx_uint16* ref_ptr = (vx_uint16*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint16* tst_ptr = (vx_uint16*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_S16:
            {
                vx_int16* ref_ptr = (vx_int16*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_int16* tst_ptr = (vx_int16*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_U32:
            {
                vx_uint32* ref_ptr = (vx_uint32*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint32* tst_ptr = (vx_uint32*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_S32:
            {
                vx_int32* ref_ptr = (vx_int32*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_int32* tst_ptr = (vx_int32*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_RGB:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                ASSERT_EQ_INT(ref_ptr[2], tst_ptr[2]);
            }
            break;

            case VX_DF_IMAGE_RGBX:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                ASSERT_EQ_INT(ref_ptr[2], tst_ptr[2]);
                ASSERT_EQ_INT(ref_ptr[3], tst_ptr[3]);
            }
            break;

            case VX_DF_IMAGE_YUYV:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);

                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // Y
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]); // U or V
            }
            break;

            case VX_DF_IMAGE_UYVY:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);

                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // U or V
                ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]); // Y
            }
            break;

            case VX_DF_IMAGE_NV12:
            case VX_DF_IMAGE_NV21:
            {
                vx_uint32 stride = (0 == plane) ? ctimg->stride : ctimg->width / 2;
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * stride + x);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);

                if (0 == plane)
                    ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]); // Y
                else
                {
                    ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
                    ASSERT_EQ_INT(ref_ptr[1], tst_ptr[1]);
                }
            }
            break;

            case VX_DF_IMAGE_IYUV:
            {
                vx_uint32 stride = (0 == plane) ? ctimg->stride : ctimg->width / 2;
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * stride / vx_addr->step_y + x / vx_addr->step_x);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            case VX_DF_IMAGE_YUV4:
            {
                vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)p_ct_base + y * ctimg->stride * ct_elem_size + x * ct_elem_size);
                vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(p_vx_base, x, y, vx_addr);
                ASSERT_EQ_INT(ref_ptr[0], tst_ptr[0]);
            }
            break;

            default:
                FAIL("unexpected image format: (%.4s)", format);
                break;
            } /* switch format */
        } /* for tst_addr.dim_x */
    } /* for tst_addr.dim_y */

    return;
} /* own_check_image_patch_plane_vx_layout() */


/* ***************************************************************************
//  vxCreateImageFromChannel tests
*/
TESTCASE(vxCreateImageFromChannel, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width;
    int height;
    vx_df_image format;
    vx_enum channel;
} CreateImageFromChannel_Arg;

#define ADD_IMAGE_FORMAT_444(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_YUV4", __VA_ARGS__, VX_DF_IMAGE_YUV4))

#define ADD_IMAGE_FORMAT_420(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_IYUV", __VA_ARGS__, VX_DF_IMAGE_IYUV)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV12", __VA_ARGS__, VX_DF_IMAGE_NV12)), \
    CT_EXPAND(nextmacro(testArgName "/VX_DF_IMAGE_NV21", __VA_ARGS__, VX_DF_IMAGE_NV21))

#define ADD_IMAGE_CHANNEL_YUV(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_CHANNEL_Y", __VA_ARGS__, VX_CHANNEL_Y)), \
    CT_EXPAND(nextmacro(testArgName "/VX_CHANNEL_U", __VA_ARGS__, VX_CHANNEL_U)), \
    CT_EXPAND(nextmacro(testArgName "/VX_CHANNEL_V", __VA_ARGS__, VX_CHANNEL_V))

#define ADD_IMAGE_CHANNEL_Y(testArgName, nextmacro, ...) \
    CT_EXPAND(nextmacro(testArgName "/VX_CHANNEL_Y", __VA_ARGS__, VX_CHANNEL_Y))

#define CREATE_IMAGE_FROM_CHANNEL_UNIFORM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("uniform", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_444, ADD_IMAGE_CHANNEL_YUV, ARG, NULL, NULL), \
    CT_GENERATE_PARAMETERS("uniform", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_420, ADD_IMAGE_CHANNEL_Y,   ARG, NULL, NULL)

#define CREATE_IMAGE_FROM_CHANNEL_RANDOM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_444, ADD_IMAGE_CHANNEL_YUV, ARG, own_generate_rand_image, NULL), \
    CT_GENERATE_PARAMETERS("rand", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_420, ADD_IMAGE_CHANNEL_Y,   ARG, own_generate_rand_image, NULL)

TEST_WITH_ARG(vxCreateImageFromChannel, testChannelFromUniformImage, CreateImageFromChannel_Arg,
    CREATE_IMAGE_FROM_CHANNEL_UNIFORM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src = 0;
    vx_image ref = 0;
    vx_image tst = 0;
    vx_uint32 width  = arg_->width;
    vx_uint32 height = arg_->height;
    vx_pixel_value_t pixel_value;

    pixel_value.YUV[0] = 0x55;
    pixel_value.YUV[1] = 0xAA;
    pixel_value.YUV[2] = 0x33;

    EXPECT_VX_OBJECT(src = vxCreateUniformImage(context, arg_->width, arg_->height, arg_->format, &pixel_value), VX_TYPE_IMAGE);

    if (VX_CHANNEL_Y != arg_->channel && VX_DF_IMAGE_IYUV == arg_->format)
    {
        width  /= 2;
        height /= 2;
    }

    EXPECT_VX_OBJECT(ref = vxCreateImage(context, width, height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    VX_CALL(vxuChannelExtract(context, src, arg_->channel, ref));

    EXPECT_VX_OBJECT(tst = vxCreateImageFromChannel(src, arg_->channel), VX_TYPE_IMAGE);

    {
        CT_Image image_ref = ct_image_from_vx_image(ref);
        CT_Image image_tst = ct_image_from_vx_image(tst);

        EXPECT_EQ_CTIMAGE(image_ref, image_tst);
    }

    VX_CALL(vxReleaseImage(&ref));
    VX_CALL(vxReleaseImage(&tst));
    VX_CALL(vxReleaseImage(&src));
} /* testChannelFromUniformImage() */

TEST_WITH_ARG(vxCreateImageFromChannel, testChannelFromRandomImage, CreateImageFromChannel_Arg,
    CREATE_IMAGE_FROM_CHANNEL_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src = 0;
    vx_image ref = 0;
    vx_image tst = 0;
    vx_uint32 width  = arg_->width;
    vx_uint32 height = arg_->height;
    CT_Image image = NULL;

    ASSERT_NO_FAILURE(image = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    EXPECT_VX_OBJECT(src = ct_image_to_vx_image(image, context), VX_TYPE_IMAGE);

    if (VX_CHANNEL_Y != arg_->channel && VX_DF_IMAGE_IYUV == arg_->format)
    {
        width  /= 2;
        height /= 2;
    }

    EXPECT_VX_OBJECT(ref = vxCreateImage(context, width, height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    VX_CALL(vxuChannelExtract(context, src, arg_->channel, ref));

    EXPECT_VX_OBJECT(tst = vxCreateImageFromChannel(src, arg_->channel), VX_TYPE_IMAGE);

    {
        /* 1. check if image created from channel is equal to channel extracted from image */
        CT_Image image_ref = ct_image_from_vx_image(ref);
        CT_Image image_tst = ct_image_from_vx_image(tst);

        EXPECT_EQ_CTIMAGE(image_ref, image_tst);
    }

    {
        /* 2. check if modification of image created from channel reflected into channel of original image */
        vx_uint32 i;
        vx_uint32 j;
        vx_uint32 p = (VX_CHANNEL_Y == arg_->channel ? 0 : (VX_CHANNEL_U == arg_->channel ? 1 : 2));
        vx_rectangle_t rect = { 1, 1, 6, 6 };
        vx_imagepatch_addressing_t addr =
        {
            rect.end_x - rect.start_x,
            rect.end_y - rect.start_y,
            1,
            rect.end_x - rect.start_x
        };

        vx_size sz = 0;
        void* ptr = 0;
        vx_map_id tst_map_id;
        vx_imagepatch_addressing_t map_addr;
        void *tst_base = NULL;
        vx_size numPixels;

        VX_CALL(vxMapImagePatch(tst, &rect, 0, &tst_map_id, &map_addr, &tst_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        numPixels = ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) *
                     ((rect.end_y-rect.start_y) * VX_SCALE_UNITY/map_addr.scale_y);
        sz = numPixels * map_addr.stride_x;
        VX_CALL(vxUnmapImagePatch(tst, tst_map_id));

        ptr = ct_alloc_mem(sz);

        /* fill image patch with some values */
        for (i = 0; i < addr.dim_y; i++)
        {
            vx_uint8* p = (vx_uint8*)ptr + i * addr.stride_x;
            for (j = 0; j < addr.dim_x; j++)
            {
                p[j] = (vx_uint8)(i + j);
            }
        }

        /* copy patch to channel image */
        vxCopyImagePatch(tst, &rect, 0, &addr, ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

        /* clean patch memory */
        ct_memset(ptr, 0, sz);

        /* get channel patch from original image */
        vxCopyImagePatch(src, &rect, p, &addr, ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

        /* check channel changes has been reflected into original image */
        for (i = 0; i < addr.dim_y; i++)
        {
            vx_uint8* p = (vx_uint8*)ptr + i * addr.stride_x;
            for (j = 0; j < addr.dim_x; j++)
            {
                EXPECT_EQ_INT((vx_uint8)(i + j), p[j]);
            }
        }

        ct_free_mem(ptr);
    }

    {
        /* 3. check if modification of channel in original image reflected into image created from channel */
        vx_uint32 i;
        vx_uint32 j;
        vx_uint32 p = (VX_CHANNEL_Y == arg_->channel ? 0 : (VX_CHANNEL_U == arg_->channel ? 1 : 2));
        vx_rectangle_t rect = { 1, 1, 6, 6 };
        vx_imagepatch_addressing_t addr =
        {
            rect.end_x - rect.start_x,
            rect.end_y - rect.start_y,
            1,
            rect.end_x - rect.start_x
        };

        vx_size sz = 0;
        void* ptr = 0;
        vx_map_id src_map_id;
        vx_imagepatch_addressing_t map_addr;
        void *src_base = NULL;
        vx_size numPixels;

        VX_CALL(vxMapImagePatch(src, &rect, p, &src_map_id, &map_addr, &src_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        numPixels = ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) *
                     ((rect.end_y-rect.start_y) * VX_SCALE_UNITY/map_addr.scale_y);
        sz = numPixels * map_addr.stride_x;
        VX_CALL(vxUnmapImagePatch(src, src_map_id));

        ptr = ct_alloc_mem(sz);

        /* fill image patch with some values */
        for (i = 0; i < addr.dim_y; i++)
        {
            vx_uint8* p = (vx_uint8*)ptr + i * addr.stride_x;
            for (j = 0; j < addr.dim_x; j++)
            {
                p[j] = (vx_uint8)(i + j);
            }
        }

        /* copy patch to channel of original image */
        vxCopyImagePatch(src, &rect, p, &addr, ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

        /* clean patch memory */
        ct_memset(ptr, 0, sz);

        /* get patch from image created from channel */
        vxCopyImagePatch(tst, &rect, 0, &addr, ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

        /* check changes of channel in original image has been reflected into channel image */
        for (i = 0; i < addr.dim_y; i++)
        {
            vx_uint8* p = (vx_uint8*)ptr + i * addr.stride_x;
            for (j = 0; j < addr.dim_x; j++)
            {
                EXPECT_EQ_INT((vx_uint8)(i + j), p[j]);
            }
        }

        ct_free_mem(ptr);
    }

    VX_CALL(vxReleaseImage(&ref));
    VX_CALL(vxReleaseImage(&tst));
    VX_CALL(vxReleaseImage(&src));
} /* testChannelFromRandomImage() */

TEST_WITH_ARG(vxCreateImageFromChannel, testChannelFromHandle, CreateImageFromChannel_Arg,
    CREATE_IMAGE_FROM_CHANNEL_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;
    vx_image src = 0;
    vx_image ref = 0;
    vx_image tst = 0;

    vx_uint32 width  = arg_->width;
    vx_uint32 height = arg_->height;

    CT_Image image = NULL;

    ASSERT_NO_FAILURE(image = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    {
        vx_uint32 n;
        vx_uint32 nplanes;

        vx_enum channel[VX_PLANE_MAX] = { VX_CHANNEL_Y, VX_CHANNEL_U, VX_CHANNEL_V, 0 };

        vx_imagepatch_addressing_t addr[VX_PLANE_MAX] =
        {
            VX_IMAGEPATCH_ADDR_INIT,
            VX_IMAGEPATCH_ADDR_INIT,
            VX_IMAGEPATCH_ADDR_INIT,
            VX_IMAGEPATCH_ADDR_INIT
        };
        void* ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

        ASSERT_NO_FAILURE(nplanes = ct_get_num_planes(arg_->format));

        for (n = 0; n < nplanes; n++)
        {
            addr[n].dim_x    = image->width  / ct_image_get_channel_subsampling_x(image, channel[n]);
            addr[n].dim_y    = image->height / ct_image_get_channel_subsampling_y(image, channel[n]);
            addr[n].stride_x = ct_image_get_channel_step_x(image, channel[n]);
            addr[n].stride_y = ct_image_get_channel_step_y(image, channel[n]);

            ptrs[n] = ct_image_get_plane_base(image, n);
        }

        EXPECT_VX_OBJECT(src = vxCreateImageFromHandle(context, arg_->format, addr, ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);
    }

    if (VX_CHANNEL_Y != arg_->channel && VX_DF_IMAGE_IYUV == arg_->format)
    {
        width  /= 2;
        height /= 2;
    }

    EXPECT_VX_OBJECT(ref = vxCreateImage(context, width, height, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    VX_CALL(vxuChannelExtract(context, src, arg_->channel, ref));

    EXPECT_VX_OBJECT(tst = vxCreateImageFromChannel(src, arg_->channel), VX_TYPE_IMAGE);

    {
        CT_Image image_ref = ct_image_from_vx_image(ref);
        CT_Image image_tst = ct_image_from_vx_image(tst);

        EXPECT_EQ_CTIMAGE(image_ref, image_tst);
    }

    VX_CALL(vxReleaseImage(&ref));
    VX_CALL(vxReleaseImage(&tst));
    VX_CALL(vxReleaseImage(&src));
} /* testChannelFromHandle() */


/* ***************************************************************************
//  vxCopyImagePatch tests
*/
TESTCASE(vxCopyImagePatch, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width;
    int height;
    vx_df_image format;
} CopyImagePatch_Arg;

#define COPY_IMAGE_PATCH_UNIFORM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("uniform",      ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   ARG, NULL, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/uniform", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, ARG, NULL, NULL)

#define COPY_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("random",      ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   ARG, own_generate_rand_image, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/random", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, ARG, own_generate_rand_image, NULL)

/*
// test vxCopyImagePatch in READ_ONLY mode from uniform image,
// independed from vxCopyImagePatch in write mode
// or vxAccessImagePatch/vxCommitImagePatch functions
*/
TEST_WITH_ARG(vxCopyImagePatch, testReadUniformImage, CopyImagePatch_Arg,
    COPY_IMAGE_PATCH_UNIFORM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    vx_pixel_value_t ref_val;

    ref_val.reserved[0] = 0x55;
    ref_val.reserved[1] = 0xaa;
    ref_val.reserved[2] = 0x33;
    ref_val.reserved[3] = 0x77;

    /* image with reference data */
    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, arg_->width, arg_->height, arg_->format, &ref_val), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        void*   ptr = 0;
        vx_size sz  = 0;
        vx_map_id image_map_id;
        vx_imagepatch_addressing_t map_addr;
        void *image_base = NULL;
        vx_size numPixels;

        VX_CALL(vxMapImagePatch(image, &rect, (vx_uint32)plane, &image_map_id, &map_addr, &image_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        numPixels = ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) *
                     ((rect.end_y-rect.start_y) * VX_SCALE_UNITY/map_addr.scale_y);
        if (map_addr.stride_x == 0 && map_addr.stride_x_bits != 0)
        {
            sz = numPixels * (map_addr.stride_x_bits *
                ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) / 8);
        }
        else
        {
            sz = numPixels * map_addr.stride_x;
        }
        VX_CALL(vxUnmapImagePatch(image, image_map_id));

        ptr = ct_alloc_mem(sz);
        ASSERT(NULL != ptr);

        addr.dim_x    = arg_->width  / own_plane_subsampling_x(arg_->format, plane);
        addr.dim_y    = arg_->height / own_plane_subsampling_y(arg_->format, plane);
        addr.stride_x = own_elem_size(arg_->format, plane);
        if (arg_->format == VX_DF_IMAGE_U1) {
            addr.stride_x_bits = 1;
            addr.stride_y = (addr.dim_x * addr.stride_x_bits + 7) / 8;
        }
        else
            addr.stride_y = addr.dim_x * addr.stride_x;

        /* read image patch */
        VX_CALL(vxCopyImagePatch(image, &rect, plane, &addr, ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        /* check if equal to reference data */
        own_check_image_patch_uniform(&ref_val, ptr, &addr, plane, arg_->format);

        ct_free_mem(ptr);
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testReadUniformImage() */

TEST_WITH_ARG(vxCopyImagePatch, testReadRandomImage, CopyImagePatch_Arg,
    COPY_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    CT_Image ref = 0;
    vx_imagepatch_addressing_t ref_addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    void* ref_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    ASSERT_NO_FAILURE(ref = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    own_image_patch_from_ct_image(ref, ref_addr, ref_ptrs, arg_->format);

    ASSERT_VX_OBJECT(image = vxCreateImageFromHandle(context, arg_->format, ref_addr, ref_ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        void* ptr = 0;
        vx_size sz = 0;
        vx_uint32 elem_size = own_elem_size(arg_->format, plane);

        vx_map_id image_map_id;
        vx_imagepatch_addressing_t map_addr;
        void *image_base = NULL;
        vx_size numPixels;

        VX_CALL(vxMapImagePatch(image, &rect, (vx_uint32)plane, &image_map_id, &map_addr, &image_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        numPixels = ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) *
                     ((rect.end_y-rect.start_y) * VX_SCALE_UNITY/map_addr.scale_y);
        if (map_addr.stride_x == 0 && map_addr.stride_x_bits != 0)
        {
            sz = numPixels * (map_addr.stride_x_bits *
                ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) / 8);
        }
        else
        {
            sz = numPixels * map_addr.stride_x;
        }
        VX_CALL(vxUnmapImagePatch(image, image_map_id));

        ptr = ct_alloc_mem(sz);
        ASSERT(NULL != ptr);

        tst_addr.dim_x    = arg_->width  / own_plane_subsampling_x(arg_->format, plane);
        tst_addr.dim_y    = arg_->height / own_plane_subsampling_y(arg_->format, plane);
        tst_addr.stride_x = elem_size;
        if (arg_->format == VX_DF_IMAGE_U1) {
            tst_addr.stride_x_bits = 1;
            tst_addr.stride_y = (tst_addr.dim_x * tst_addr.stride_x_bits + 7) / 8;
        }
        else
            tst_addr.stride_y = tst_addr.dim_x * tst_addr.stride_x;

        VX_CALL(vxCopyImagePatch(image, &rect, plane, &tst_addr, ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        /* check if image patch plane equal to reference data */
        own_check_image_patch_plane_user_layout(ref, &tst_addr, ptr, plane, arg_->format);

        ct_free_mem(ptr);
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testReadRandomImage() */

TEST_WITH_ARG(vxCopyImagePatch, testWriteRandomImage, CopyImagePatch_Arg,
    COPY_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    CT_Image ref = 0;
    vx_rectangle_t rect = { 0, 0, arg_->width, arg_->height };
    vx_imagepatch_addressing_t ref_addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    void* ref_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    ASSERT_NO_FAILURE(ref = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    ASSERT_VX_OBJECT(image = vxCreateImage(context, arg_->width, arg_->height, arg_->format), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    own_image_patch_from_ct_image(ref, ref_addr, ref_ptrs, arg_->format);

    /* write reference data to image */
    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        VX_CALL(vxCopyImagePatch(image, &rect, plane, &ref_addr[plane], ref_ptrs[plane], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
    }

    /* get data from image to external memory to compare with reference */
    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect     = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        void* ptr = 0;
        vx_size sz = 0;
        vx_uint32 elem_size = own_elem_size(arg_->format, plane);

        vx_map_id image_map_id;
        vx_imagepatch_addressing_t map_addr;
        void *image_base = NULL;
        vx_size numPixels;

        VX_CALL(vxMapImagePatch(image, &rect, (vx_uint32)plane, &image_map_id, &map_addr, &image_base, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
        numPixels = ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) *
                     ((rect.end_y-rect.start_y) * VX_SCALE_UNITY/map_addr.scale_y);
        if (map_addr.stride_x == 0 && map_addr.stride_x_bits != 0)
        {
            sz = numPixels * (map_addr.stride_x_bits *
                ((rect.end_x-rect.start_x) * VX_SCALE_UNITY/map_addr.scale_x) / 8);
        }
        else
        {
            sz = numPixels * map_addr.stride_x;
        }
        VX_CALL(vxUnmapImagePatch(image, image_map_id));

        ptr = ct_alloc_mem(sz);
        ASSERT(NULL != ptr);

        tst_addr.dim_x    = arg_->width  / own_plane_subsampling_x(arg_->format, plane);
        tst_addr.dim_y    = arg_->height / own_plane_subsampling_y(arg_->format, plane);
        tst_addr.stride_x = elem_size;
        if (arg_->format == VX_DF_IMAGE_U1) {
            tst_addr.stride_x_bits = 1;
            tst_addr.stride_y = (tst_addr.dim_x * tst_addr.stride_x_bits + 7) / 8;
        }
        else
            tst_addr.stride_y = tst_addr.dim_x * tst_addr.stride_x;

        VX_CALL(vxCopyImagePatch(image, &rect, plane, &tst_addr, ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        /* check if image patch plane equal to reference data */
        own_check_image_patch_plane_user_layout(ref, &tst_addr, ptr, plane, arg_->format);

        ct_free_mem(ptr);
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testWriteRandomImage() */


/* ***************************************************************************
//  vxMapImagePatch tests
*/
TESTCASE(vxMapImagePatch, CT_VXContext, ct_setup_vx_context, 0)

typedef struct
{
    const char* testName;
    CT_Image(*generator)(const char* fileName, int width, int height, vx_df_image format);
    const char* fileName;
    int width;
    int height;
    vx_df_image format;
} MapImagePatch_Arg;

#define MAP_IMAGE_PATCH_UNIFORM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("uniform",      ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   ARG, NULL, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/uniform", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, ARG, NULL, NULL)

#define MAP_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS \
    CT_GENERATE_PARAMETERS("random",      ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMATS,   ARG, own_generate_rand_image, NULL), \
    CT_GENERATE_PARAMETERS("_U1_/random", ADD_SIZE_SMALL_SET, ADD_IMAGE_FORMAT_U1, ARG, own_generate_rand_image, NULL)

/*
// test vxMapImagePatch in READ_ONLY mode from uniform image,
// independed from vxMapImagePatch/vxCopyImagePatch in write mode
// or vxAccessImagePatch/vxCommitImagePatch functions
*/
TEST_WITH_ARG(vxMapImagePatch, testMapReadUniformImage, MapImagePatch_Arg,
    MAP_IMAGE_PATCH_UNIFORM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 plane;
    vx_size  num_planes = 0;
    vx_image image = 0;
    vx_pixel_value_t ref_val;

    ref_val.reserved[0] = 0x55;
    ref_val.reserved[1] = 0xaa;
    ref_val.reserved[2] = 0x33;
    ref_val.reserved[3] = 0x77;

    ASSERT_VX_OBJECT(image = vxCreateUniformImage(context, arg_->width, arg_->height, arg_->format, &ref_val), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void*   ptr = 0;

        VX_CALL(vxMapImagePatch(image, &rect, (vx_uint32)plane, &map_id, &addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, flags));

        for (y = 0; y < addr.dim_y; y += addr.step_y)
        {
            for (x = 0; x < addr.dim_x; x += addr.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U1:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.U1 ? 1 : 0, (tst[0] & (1 << (x % 8))) >> (x % 8));
                }
                break;

                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.reserved[0], tst[0]);
                }
                break;

                case VX_DF_IMAGE_U16:
                {
                    vx_uint16* tst = (vx_uint16*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.U16, tst[0]);
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* tst = (vx_int16*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.S16, tst[0]);
                }
                break;

                case VX_DF_IMAGE_U32:
                {
                    vx_uint32* tst = (vx_uint32*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.U32, tst[0]);
                }
                break;

                case VX_DF_IMAGE_S32:
                {
                    vx_int32* tst = (vx_int32*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.S32, tst[0]);
                }
                break;

                case VX_DF_IMAGE_RGB:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.RGB[0], tst[0]);
                    ASSERT_EQ_INT(ref_val.RGB[1], tst[1]);
                    ASSERT_EQ_INT(ref_val.RGB[2], tst[2]);
                }
                break;

                case VX_DF_IMAGE_RGBX:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.RGBX[0], tst[0]);
                    ASSERT_EQ_INT(ref_val.RGBX[1], tst[1]);
                    ASSERT_EQ_INT(ref_val.RGBX[2], tst[2]);
                    ASSERT_EQ_INT(ref_val.RGBX[3], tst[3]);
                }
                break;

                case VX_DF_IMAGE_YUYV:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);

                    vx_uint8 tst_u_or_v = tst[1];

                    ASSERT_EQ_INT(ref_val.YUV[0], tst[0]); // Y

                    if (x & 1)
                        ASSERT_EQ_INT(ref_val.YUV[2], tst_u_or_v); // V
                    else
                        ASSERT_EQ_INT(ref_val.YUV[1], tst_u_or_v); // U
                }
                break;

                case VX_DF_IMAGE_UYVY:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);

                    vx_uint8 tst_u_or_v = tst[0];

                    ASSERT_EQ_INT(ref_val.YUV[0], tst[1]); // Y

                    if (x & 1)
                        ASSERT_EQ_INT(ref_val.YUV[2], tst_u_or_v); // V
                    else
                        ASSERT_EQ_INT(ref_val.YUV[1], tst_u_or_v); // U
                }
                break;

                case VX_DF_IMAGE_NV12:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);

                    if (0 == plane)
                        ASSERT_EQ_INT(ref_val.YUV[0], tst[0]); // Y
                    else
                    {
                        ASSERT_EQ_INT(ref_val.YUV[1], tst[0]); // U
                        ASSERT_EQ_INT(ref_val.YUV[2], tst[1]); // V
                    }
                }
                break;

                case VX_DF_IMAGE_NV21:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);

                    if (0 == plane)
                        ASSERT_EQ_INT(ref_val.YUV[0], tst[0]); // Y
                    else
                    {
                        ASSERT_EQ_INT(ref_val.YUV[1], tst[1]); // U
                        ASSERT_EQ_INT(ref_val.YUV[2], tst[0]); // V
                    }
                }
                break;

                case VX_DF_IMAGE_YUV4:
                case VX_DF_IMAGE_IYUV:
                {
                    vx_uint8* tst = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &addr);
                    ASSERT_EQ_INT(ref_val.YUV[plane], tst[0]);
                }
                break;

                default:
                    FAIL("unexpected image format: (%.4s)", arg_->format);
                break;
                } /* switch format */
            } /* for addr.dim_x */
        } /* for addr.dim_y */

        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testMapReadUniformImage() */

TEST_WITH_ARG(vxMapImagePatch, testMapReadRandomImage, MapImagePatch_Arg,
    MAP_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    CT_Image ref = 0;
    vx_imagepatch_addressing_t ref_addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    void* ref_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    ASSERT_NO_FAILURE(ref = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    own_image_patch_from_ct_image(ref, ref_addr, ref_ptrs, arg_->format);

    ASSERT_VX_OBJECT(image = vxCreateImageFromHandle(context, arg_->format, ref_addr, ref_ptrs, VX_MEMORY_TYPE_HOST), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect     = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr = 0;

        VX_CALL(vxMapImagePatch(image, &rect, plane, &map_id, &tst_addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, flags));

        /* check if image patch plane equal to reference data */
        own_check_image_patch_plane_vx_layout(ref, &tst_addr, ptr, plane, arg_->format);

        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testMapReadRandomImage() */

TEST_WITH_ARG(vxMapImagePatch, testMapReadWriteRandomImage, MapImagePatch_Arg,
    MAP_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    CT_Image ref = 0;
    vx_imagepatch_addressing_t ref_addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    void* ref_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    /* generate random reference data */
    ASSERT_NO_FAILURE(ref = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    /* image patch info for generated data */
    own_image_patch_from_ct_image(ref, ref_addr, ref_ptrs, arg_->format);

    /* image to test with generated reference data */
    ASSERT_VX_OBJECT(image = vxCreateImage(context, arg_->width, arg_->height, arg_->format), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr = 0;

        /* fill image with generated data from image patch */
        VX_CALL(vxCopyImagePatch(image, &rect, plane, &ref_addr[plane], ref_ptrs[plane], VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

        /* map image patch for read and write */
        VX_CALL(vxMapImagePatch(image, &rect, plane, &map_id, &tst_addr, &ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, flags));

        /* check if image patch plane equal to reference data */
        own_check_image_patch_plane_vx_layout(ref, &tst_addr, ptr, plane, arg_->format);

        /* modify mapped image patch data (i.e. increment with saturation each pixel value) */
        for (y = 0; y < tst_addr.dim_y; y += tst_addr.step_y)
        {
            for (x = 0; x < tst_addr.dim_x; x += tst_addr.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U1:
                {
                    vx_uint8 offset = x % 8;
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = ( tst_ptr[0] & ~(1 << offset)) |
                                 (~tst_ptr[0] &  (1 << offset));
                }
                break;

                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_U16:
                {
                    vx_uint16* tst_ptr = (vx_uint16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint16)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* tst_ptr = (vx_int16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_int16)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_U32:
                {
                    vx_uint32* tst_ptr = (vx_uint32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint32)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_S32:
                {
                    vx_int32* tst_ptr = (vx_int32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_int32)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_RGB:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(tst_ptr[1] + 1);
                    tst_ptr[2] = (vx_uint8)(tst_ptr[2] + 1);
                }
                break;

                case VX_DF_IMAGE_RGBX:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(tst_ptr[1] + 1);
                    tst_ptr[2] = (vx_uint8)(tst_ptr[2] + 1);
                    tst_ptr[3] = (vx_uint8)(tst_ptr[3] + 1);
                }
                break;

                case VX_DF_IMAGE_YUYV:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(tst_ptr[1] + 1);
                }
                break;

                case VX_DF_IMAGE_UYVY:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(tst_ptr[1] + 1);
                }
                break;

                case VX_DF_IMAGE_NV12:
                case VX_DF_IMAGE_NV21:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    if (0 == plane)
                        tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                    else
                    {
                        tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                        tst_ptr[1] = (vx_uint8)(tst_ptr[1] + 1);
                    }
                }
                break;

                case VX_DF_IMAGE_IYUV:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_YUV4:
                {
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(tst_ptr[0] + 1);
                }
                break;

                default:
                    FAIL("unexpected image format: (%.4s)", arg_->format);
                break;
                } /* switch format */
            } /* for tst_addr.dim_x */
        } /* for tst_addr.dim_y */

        /* commit modified image patch */
        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    /* check if modified image patch was actually commited into image */
    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr = 0;
        vx_uint32 elem_size = own_elem_size(arg_->format, plane);

        /* map image patch for read */
        VX_CALL(vxMapImagePatch(image, &rect, (vx_uint32)plane, &map_id, &tst_addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, flags));

        /* check if pixel values are incremented */
        for (y = 0; y < tst_addr.dim_y; y += tst_addr.step_y)
        {
            for (x = 0; x < tst_addr.dim_x; x += tst_addr.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U1:
                {
                    vx_uint8  offset  = x % 8;
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ct_stride_bytes(ref) +
                                                    (x * ct_image_bits_per_pixel(VX_DF_IMAGE_U1)) / 8);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(( ref_ptr[0] & (1 << offset)) >> offset,
                                  (~tst_ptr[0] & (1 << offset)) >> offset);
                }
                break;

                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_U16:
                {
                    vx_uint16* ref_ptr = (vx_uint16*)((vx_uint8*)ref->data.u16 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint16* tst_ptr = (vx_uint16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint16)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* ref_ptr = (vx_int16*)((vx_uint8*)ref->data.s16 + y * ref->stride * elem_size + x * elem_size);
                    vx_int16* tst_ptr = (vx_int16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_int16)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_U32:
                {
                    vx_uint32* ref_ptr = (vx_uint32*)((vx_uint8*)ref->data.u32 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint32* tst_ptr = (vx_uint32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint32)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_S32:
                {
                    vx_int32* ref_ptr = (vx_int32*)((vx_uint8*)ref->data.s32 + y * ref->stride * elem_size + x * elem_size);
                    vx_int32* tst_ptr = (vx_int32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_int32)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_RGB:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgb + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    ASSERT_EQ_INT(ref_ptr[2], (vx_uint8)(tst_ptr[2] - 1));
                }
                break;

                case VX_DF_IMAGE_RGBX:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgbx + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    ASSERT_EQ_INT(ref_ptr[2], (vx_uint8)(tst_ptr[2] - 1));
                    ASSERT_EQ_INT(ref_ptr[3], (vx_uint8)(tst_ptr[3] - 1));
                }
                break;

                case VX_DF_IMAGE_YUYV:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.yuyv + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // Y
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1)); // U or V
                }
                break;

                case VX_DF_IMAGE_UYVY:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.uyvy + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // U or V
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1)); // Y
                }
                break;

                case VX_DF_IMAGE_NV12:
                case VX_DF_IMAGE_NV21:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride + x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    if (0 == plane)
                        ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // Y
                    else
                    {
                        ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                        ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    }
                }
                break;

                case VX_DF_IMAGE_IYUV:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride / tst_addr.step_y + x / tst_addr.step_x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_YUV4:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                default:
                    FAIL("unexpected image format: (%.4s)", arg_->format);
                break;
                } /* switch format */
            } /* for tst_addr.dim_x */
        } /* for tst_addr.dim_y */

        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testMapReadWriteRandomImage() */

TEST_WITH_ARG(vxMapImagePatch, testMapWriteRandomImage, MapImagePatch_Arg,
    MAP_IMAGE_PATCH_RANDOM_IMAGE_PARAMETERS
)
{
    vx_context context = context_->vx_context_;

    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 plane;
    vx_size num_planes = 0;
    vx_image image = 0;
    CT_Image ref = 0;
    vx_imagepatch_addressing_t ref_addr[VX_PLANE_MAX] =
    {
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT,
        VX_IMAGEPATCH_ADDR_INIT
    };
    void* ref_ptrs[VX_PLANE_MAX] = { 0, 0, 0, 0 };

    /* generate random reference data */
    ASSERT_NO_FAILURE(ref = arg_->generator(arg_->fileName, arg_->width, arg_->height, arg_->format));

    /* image patch info for generated data */
    own_image_patch_from_ct_image(ref, ref_addr, ref_ptrs, arg_->format);

    /* image to test with generated reference data */
    ASSERT_VX_OBJECT(image = vxCreateImage(context, arg_->width, arg_->height, arg_->format), VX_TYPE_IMAGE);

    VX_CALL(vxQueryImage(image, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr = 0;
        vx_uint32 elem_size = own_elem_size(arg_->format, plane);

        /* map image patch for write */
        VX_CALL(vxMapImagePatch(image, &rect, plane, &map_id, &tst_addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, flags));

        /* modify mapped image patch (write incremented reference pixel values) */
        for (y = 0; y < tst_addr.dim_y; y += tst_addr.step_y)
        {
            for (x = 0; x < tst_addr.dim_x; x += tst_addr.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U1:
                {
                    vx_uint8  offset  = x % 8;
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ct_stride_bytes(ref) +
                                                    (x * ct_image_bits_per_pixel(VX_DF_IMAGE_U1)) / 8);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (( tst_ptr[0] & ~(1 << offset)) |
                                  (~ref_ptr[0] &  (1 << offset)));
                }
                break;

                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_U16:
                {
                    vx_uint16* ref_ptr = (vx_uint16*)((vx_uint8*)ref->data.u16 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint16* tst_ptr = (vx_uint16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint16)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* ref_ptr = (vx_int16*)((vx_uint8*)ref->data.s16 + y * ref->stride * elem_size + x * elem_size);
                    vx_int16* tst_ptr = (vx_int16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_int16)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_U32:
                {
                    vx_uint32* ref_ptr = (vx_uint32*)((vx_uint8*)ref->data.u32 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint32* tst_ptr = (vx_uint32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint32)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_S32:
                {
                    vx_int32* ref_ptr = (vx_int32*)((vx_uint8*)ref->data.s32 + y * ref->stride * elem_size + x * elem_size);
                    vx_int32* tst_ptr = (vx_int32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_int32)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_RGB:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgb + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(ref_ptr[1] + 1);
                    tst_ptr[2] = (vx_uint8)(ref_ptr[2] + 1);
                }
                break;

                case VX_DF_IMAGE_RGBX:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgbx + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(ref_ptr[1] + 1);
                    tst_ptr[2] = (vx_uint8)(ref_ptr[2] + 1);
                    tst_ptr[3] = (vx_uint8)(ref_ptr[3] + 1);
                }
                break;

                case VX_DF_IMAGE_YUYV:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.yuyv + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(ref_ptr[1] + 1);
                }
                break;

                case VX_DF_IMAGE_UYVY:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.uyvy + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                    tst_ptr[1] = (vx_uint8)(ref_ptr[1] + 1);
                }
                break;

                case VX_DF_IMAGE_NV12:
                case VX_DF_IMAGE_NV21:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride + x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    if (0 == plane)
                        tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                    else
                    {
                        tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                        tst_ptr[1] = (vx_uint8)(ref_ptr[1] + 1);
                    }
                }
                break;

                case VX_DF_IMAGE_IYUV:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride / tst_addr.step_y + x / tst_addr.step_x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                }
                break;

                case VX_DF_IMAGE_YUV4:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    tst_ptr[0] = (vx_uint8)(ref_ptr[0] + 1);
                }
                break;

                default:
                    FAIL("unexpected image format: (%.4s)", arg_->format);
                break;
                } /* switch format */
            } /* for tst_addr.dim_x */
        } /* for tst_addr.dim_y */

        /* commit modified image patch */
        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    /* check if modified image patch was actually commited into image */
    for (plane = 0; plane < (vx_uint32)num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, arg_->width, arg_->height };
        vx_imagepatch_addressing_t tst_addr = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr = 0;
        vx_uint32 elem_size = own_elem_size(arg_->format, plane);

        /* map image patch for read */
        VX_CALL(vxMapImagePatch(image, &rect, plane, &map_id, &tst_addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, flags));

        /* check if pixel values are incremented */
        for (y = 0; y < tst_addr.dim_y; y += tst_addr.step_y)
        {
            for (x = 0; x < tst_addr.dim_x; x += tst_addr.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U1:
                {
                    vx_uint8  offset  = x % 8;
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ct_stride_bytes(ref) +
                                                    (x * ct_image_bits_per_pixel(VX_DF_IMAGE_U1)) / 8);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(( ref_ptr[0] & (1 << offset)) >> offset,
                                  (~tst_ptr[0] & (1 << offset)) >> offset);
                }
                break;

                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.y + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_U16:
                {
                    vx_uint16* ref_ptr = (vx_uint16*)((vx_uint8*)ref->data.u16 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint16* tst_ptr = (vx_uint16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint16)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* ref_ptr = (vx_int16*)((vx_uint8*)ref->data.s16 + y * ref->stride * elem_size + x * elem_size);
                    vx_int16* tst_ptr = (vx_int16*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_int16)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_U32:
                {
                    vx_uint32* ref_ptr = (vx_uint32*)((vx_uint8*)ref->data.u32 + y * ref->stride * elem_size + x * elem_size);
                    vx_uint32* tst_ptr = (vx_uint32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint32)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_S32:
                {
                    vx_int32* ref_ptr = (vx_int32*)((vx_uint8*)ref->data.s32 + y * ref->stride * elem_size + x * elem_size);
                    vx_int32* tst_ptr = (vx_int32*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_int32)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_RGB:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgb + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    ASSERT_EQ_INT(ref_ptr[2], (vx_uint8)(tst_ptr[2] - 1));
                }
                break;

                case VX_DF_IMAGE_RGBX:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.rgbx + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    ASSERT_EQ_INT(ref_ptr[2], (vx_uint8)(tst_ptr[2] - 1));
                    ASSERT_EQ_INT(ref_ptr[3], (vx_uint8)(tst_ptr[3] - 1));
                }
                break;

                case VX_DF_IMAGE_YUYV:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.yuyv + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // Y
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1)); // U or V
                }
                break;

                case VX_DF_IMAGE_UYVY:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)((vx_uint8*)ref->data.uyvy + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // U or V
                    ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1)); // Y
                }
                break;

                case VX_DF_IMAGE_NV12:
                case VX_DF_IMAGE_NV21:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride + x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);

                    if (0 == plane)
                        ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1)); // Y
                    else
                    {
                        ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                        ASSERT_EQ_INT(ref_ptr[1], (vx_uint8)(tst_ptr[1] - 1));
                    }
                }
                break;

                case VX_DF_IMAGE_IYUV:
                {
                    vx_uint32 stride = (0 == plane) ? ref->stride : ref->width / 2;
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * stride / tst_addr.step_y + x / tst_addr.step_x);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                case VX_DF_IMAGE_YUV4:
                {
                    vx_uint8* ref_ptr = (vx_uint8*)(ct_image_get_plane_base(ref, plane) + y * ref->stride * elem_size + x * elem_size);
                    vx_uint8* tst_ptr = (vx_uint8*)vxFormatImagePatchAddress2d(ptr, x, y, &tst_addr);
                    ASSERT_EQ_INT(ref_ptr[0], (vx_uint8)(tst_ptr[0] - 1));
                }
                break;

                default:
                    FAIL("unexpected image format: (%.4s)", arg_->format);
                break;
                } /* switch format */
            } /* for tst_addr.dim_x */
        } /* for tst_addr.dim_y */

        VX_CALL(vxUnmapImagePatch(image, map_id));
    } /* for num_planes */

    VX_CALL(vxReleaseImage(&image));
    ASSERT(image == 0);

    return;
} /* testMapWriteRandomImage() */

TESTCASE_TESTS(Image,
    testRngImageCreation,
    testImageCreation_U1,
    testVirtualImageCreation,
    testVirtualImageCreationDims,
    testCreateImageFromHandle,
    testSwapImageHandle,
    testFormatImagePatchAddress1d,
    testConvert_CT_Image,
    testvxSetImagePixelValues,
    testUniformImage,
    DISABLED_testAccessCopyWrite,
    DISABLED_testAccessCopyRead,
    DISABLED_testAccessCopyWriteUniformImage,
    testQueryImage
)

TESTCASE_TESTS(vxCreateImageFromChannel,
    testChannelFromUniformImage,
    testChannelFromRandomImage,
    testChannelFromHandle
)

TESTCASE_TESTS(vxCopyImagePatch,
    testReadUniformImage,
    testReadRandomImage,
    testWriteRandomImage
)

TESTCASE_TESTS(vxMapImagePatch,
    testMapReadUniformImage,
    testMapReadRandomImage,
    testMapReadWriteRandomImage,
    testMapWriteRandomImage
)

#endif //OPENVX_USE_ENHANCED_VISION || OPENVX_CONFORMANCE_VISION
