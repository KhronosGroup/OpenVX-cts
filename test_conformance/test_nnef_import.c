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

#ifdef OPENVX_CONFORMANCE_NNEF_IMPORT

#include <VX/vx_khr_import_kernel.h>

#include "test_engine/test.h"

typedef struct {
    const char* name;
    char  *type;
    char  *url;
} nnef_import_arg;

#define NNEF_IMPORT_PARAMETERS \
    ARG("importkernel", "vx_kernel", "./kernel.img") \


TESTCASE(TensorNNEFImport, CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG(TensorNNEFImport, testNNEFImport, nnef_import_arg,
    NNEF_IMPORT_PARAMETERS)
{
    vx_context context = context_->vx_context_;
    vx_char *type = arg_->type;
    vx_char *url = arg_->url;
    vx_kernel kernel = NULL;
    vx_status status = VX_SUCCESS;

    kernel = vxImportKernelFromURL(context, type, url);

    status = vxGetStatus((vx_reference)kernel);

    if(VX_SUCCESS == status)
    {
        VX_CALL(vxReleaseKernel(&kernel));
        ASSERT(kernel == 0);
    }
}

TESTCASE_TESTS(TensorNNEFImport, testNNEFImport)
#endif
