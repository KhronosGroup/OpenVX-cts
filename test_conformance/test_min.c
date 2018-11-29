/*

 * Copyright (c) 2017-2017 The Khronos Group Inc.
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

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"


TESTCASE(Min, CT_VXContext, ct_setup_vx_context, 0)

typedef struct {
    const char* name;
    int mode;
    vx_df_image format;
} format_arg;

#define MIN_TEST_CASE(imm, tp) \
    {#imm "/" #tp, CT_##imm##_MODE, VX_DF_IMAGE_##tp}

TEST_WITH_ARG(Min, testvxMin, format_arg,
              MIN_TEST_CASE(Immediate, U8),
              MIN_TEST_CASE(Graph, U8),
              MIN_TEST_CASE(Immediate, S16),
              MIN_TEST_CASE(Graph, S16),
              )
{
    int format = arg_->format;
    int mode = arg_->mode;
    vx_image src_in0;
    vx_image src_in1;
    vx_image out,ref;
    vx_graph graph = 0;
    vx_node node = 0;
    vx_context context = context_->vx_context_;

    vx_pixel_value_t vals0;

    vals0.reserved[0] = 0x10;
    vals0.reserved[1] = 0x11;
    vals0.reserved[2] = 0x12;
    vals0.reserved[3] = 0x13;
    vx_pixel_value_t vals1;

    vals1.reserved[0] = 0x11;
    vals1.reserved[1] = 0x11;
    vals1.reserved[2] = 0x12;
    vals1.reserved[3] = 0x13;

    CT_Image output = NULL;
    ASSERT_VX_OBJECT(src_in0 = vxCreateImage(context, 640, 480, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(src_in1 = vxCreateImage(context, 640, 480, format), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(ref = vxCreateUniformImage(context, 640, 480, format, &vals0), VX_TYPE_IMAGE);

    vx_uint32 x;
    vx_uint32 y;
    vx_uint32 plane;
    vx_size num_planes = 0;
    VX_CALL(vxQueryImage(src_in0, VX_IMAGE_PLANES, &num_planes, sizeof(num_planes)));

    for (plane = 0; plane < num_planes; plane++)
    {
        vx_rectangle_t             rect = { 0, 0, 640, 480 };
        vx_imagepatch_addressing_t tst_addr0 = VX_IMAGEPATCH_ADDR_INIT;
        vx_imagepatch_addressing_t tst_addr1 = VX_IMAGEPATCH_ADDR_INIT;
        vx_map_id map_id0;
        vx_map_id map_id1;
        vx_uint32 flags = VX_NOGAP_X;
        void* ptr0 = 0;
        void* ptr1 = 0;

        VX_CALL(vxMapImagePatch(src_in0, &rect, plane, &map_id0, &tst_addr0, &ptr0, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, flags));
        VX_CALL(vxMapImagePatch(src_in1, &rect, plane, &map_id1, &tst_addr1, &ptr1, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, flags));

        for (y = 0; y < tst_addr0.dim_y; y += tst_addr0.step_y)
        {
            for (x = 0; x < tst_addr0.dim_x; x += tst_addr0.step_x)
            {
                switch (arg_->format)
                {
                case VX_DF_IMAGE_U8:
                {
                    vx_uint8* tst_ptr0 = (vx_uint8*)vxFormatImagePatchAddress2d(ptr0, x, y, &tst_addr0);
                    if(x >= y)
                        tst_ptr0[0] = vals1.U8;
                    else
                        tst_ptr0[0] = vals0.U8;
                }
                break;

                case VX_DF_IMAGE_S16:
                {
                    vx_int16* tst_ptr0 = (vx_int16*)vxFormatImagePatchAddress2d(ptr0, x, y, &tst_addr0);
                    if(x >= y)
                        tst_ptr0[0] = vals1.S16;
                    else
                        tst_ptr0[0] = vals0.S16;
                }
                break;
               }
            }
        }
        VX_CALL(vxUnmapImagePatch(src_in0, map_id0));

        for (y = 0; y < tst_addr1.dim_y; y += tst_addr1.step_y)
        {
            for (x = 0; x < tst_addr1.dim_x; x += tst_addr1.step_x)
            {
                switch (arg_->format)
                {
                    case VX_DF_IMAGE_U8:
                    {
                        vx_uint8* tst_ptr1 = (vx_uint8*)vxFormatImagePatchAddress2d(ptr1, x, y, &tst_addr1);
                        if (x >= y)
                            tst_ptr1[0] = vals0.U8;
                        else
                            tst_ptr1[0] = vals1.U8;
                    }
                    break;

                    case VX_DF_IMAGE_S16:
                    {
                        vx_int16* tst_ptr1 = (vx_int16*)vxFormatImagePatchAddress2d(ptr1, x, y, &tst_addr1);
                        if (x >= y)
                            tst_ptr1[0] = vals0.S16;
                        else
                            tst_ptr1[0] = vals1.S16;
                    }
                    break;
               }
            }
        }
        VX_CALL(vxUnmapImagePatch(src_in1, map_id1));
    }

    ASSERT_NO_FAILURE(output = ct_allocate_image(640, 480, format));
    ASSERT_VX_OBJECT(out = ct_image_to_vx_image(output, context), VX_TYPE_IMAGE);

    if( mode == CT_Immediate_MODE )
    {
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, vxuMin(context, src_in0, src_in1, out));
    }
    else
    {
        graph = vxCreateGraph(context);
        ASSERT_VX_OBJECT(graph, VX_TYPE_GRAPH);
        node = vxMinNode(graph, src_in0, src_in1, out);
        ASSERT_VX_OBJECT(node, VX_TYPE_NODE);
        VX_CALL(vxVerifyGraph(graph));
        VX_CALL(vxProcessGraph(graph));
    }

    CT_Image image_ref = ct_image_from_vx_image(ref);
    CT_Image image_out = ct_image_from_vx_image(out);

    EXPECT_EQ_CTIMAGE(image_ref, image_out);

    if(node)
        VX_CALL(vxReleaseNode(&node));
    if(graph)
        VX_CALL(vxReleaseGraph(&graph));
    ASSERT(node == 0 && graph == 0);
    VX_CALL(vxReleaseImage(&src_in0));
    VX_CALL(vxReleaseImage(&src_in1));
    VX_CALL(vxReleaseImage(&out));
    VX_CALL(vxReleaseImage(&ref));

}
TESTCASE_TESTS(Min, testvxMin)
