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

#include <string.h>
#include <VX/vx.h>
#include <VX/vxu.h>

#include "test_engine/test.h"

TESTCASE(Distribution, CT_VXContext, ct_setup_vx_context, 0)

static vx_uint32 reference_window(vx_uint32 range, vx_size nbins)
{
    vx_uint32 test_window = (vx_uint32)(range / nbins);
    if (test_window*nbins == range)
        return test_window;
    else
        return 0;
}

#define MAX_BINS 256

TEST(Distribution, testvxCreateVirtualDistribution)
{
    vx_distribution dist1;
    vx_context context = context_->vx_context_;
    uint64_t rng;
    rng = CT()->seed_;
    int val0 = CT_RNG_NEXT_INT(rng, 0, (MAX_BINS-1)), val1 = CT_RNG_NEXT_INT(rng, 0, (MAX_BINS-1));
    int offset = CT_MIN(val0, val1), range = CT_MAX(val0, val1) - offset + 1;
    int nbins = CT_RNG_NEXT_INT(rng, 1, range+1);
    vx_graph graph = 0;
    graph = vxCreateGraph(context);

    dist1 = vxCreateVirtualDistribution(graph, nbins, offset, range);
    ASSERT_VX_OBJECT(dist1, VX_TYPE_DISTRIBUTION);

    {
        /* smoke tests for query distribution attributes */
        vx_size   attr_dims = 0;
        vx_int32  attr_offset = 0;
        vx_uint32 attr_range = 0;
        vx_size   attr_bins = 0;
        vx_uint32 attr_window = 0;
        VX_CALL(vxQueryDistribution(dist1, VX_DISTRIBUTION_DIMENSIONS, &attr_dims, sizeof(attr_dims)));
        if (1 != attr_dims)
            CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_DIMENSIONS failed\n");

        VX_CALL(vxQueryDistribution(dist1, VX_DISTRIBUTION_OFFSET, &attr_offset, sizeof(attr_offset)));
        if (attr_offset != offset)
            CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_OFFSET failed\n");

        VX_CALL(vxQueryDistribution(dist1, VX_DISTRIBUTION_RANGE, &attr_range, sizeof(attr_range)));
        if (attr_range != range)
            CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_RANGE failed\n");

        VX_CALL(vxQueryDistribution(dist1, VX_DISTRIBUTION_BINS, &attr_bins, sizeof(attr_bins)));
        if (attr_bins != nbins)
            CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_BINS failed\n");

        VX_CALL(vxQueryDistribution(dist1, VX_DISTRIBUTION_WINDOW, &attr_window, sizeof(attr_window)));
        /*Tthe attribute is specified as valid only when the range is a multiple of nbins, 
        * in other cases, its value shouldn't be checked */
        if (((range % nbins) == 0) && (attr_window != reference_window(range, nbins)))
            CT_FAIL("check for query distribution attribute VX_DISTRIBUTION_WINDOW failed\n");

    }
    VX_CALL(vxReleaseDistribution(&dist1));
    VX_CALL(vxReleaseGraph(&graph));
    ASSERT(dist1 == 0 && graph == 0);
}

TESTCASE_TESTS(Distribution, testvxCreateVirtualDistribution)
