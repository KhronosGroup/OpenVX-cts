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

#include "nnef_parser.h"

#include <string.h>

#define MAXLEN 512
#define MAX_TENSORS 64
#define DIFF_EPSILON 10e-4

typedef struct {
    vx_char *name;
    vx_char *type;
    vx_char *url;
} nnef_import_arg;

static vx_char file_names[MAX_NNEF_KERNELS][MAXPATHLENGTH];
static vx_int32 file_num = 0;
static vx_char nnef_kernel_url[MAX_NNEF_KERNELS][MAXPATHLENGTH];

static inline void *mem_init(vx_int32 size)
{
    void *ptr = NULL;

    ptr = ct_alloc_mem(size);

    memset(ptr, 0, size);

    return ptr;
}

static inline void free_tensors_paramters(vx_int32 *output_tensor_type, vx_size *output_num_dims, vx_size **output_dims,
                                          vx_char **output_data, vx_size **output_tensor_strides, vx_size **output_view_start,
                                          vx_char **filenames, vx_int32 num_params, vx_int32 output_num)
{
    vx_int32 i;

    for (i = 0; i < num_params; i++)
        ct_free_mem(filenames[i]);

    for (i = 0; i < output_num; i++)
    {
        ct_free_mem(output_data[i]);
        ct_free_mem(output_dims[i]);
        ct_free_mem(output_tensor_strides[i]);
        ct_free_mem(output_view_start[i]);
    }

    ct_free_mem(filenames);
    ct_free_mem(output_tensor_type);
    ct_free_mem(output_num_dims);
    ct_free_mem(output_dims);
    ct_free_mem(output_data);
    ct_free_mem(output_tensor_strides);
    ct_free_mem(output_view_start);
}


static inline void query_input_output_num(vx_kernel nn_kernel, vx_int32 num_params, vx_int32 *input_num, vx_int32 *output_num)
{
    vx_int32 direction, i;

    for (i = 0; i < num_params; i++)
    {
        vx_parameter prm = vxGetKernelParameterByIndex(nn_kernel, i);

        vxQueryParameter(prm, VX_PARAMETER_DIRECTION, &direction, sizeof(enum vx_type_e));

        if (direction == VX_INPUT)
            (*input_num)++;
        else if (direction == VX_OUTPUT)
            (*output_num)++;

        VX_CALL(vxReleaseParameter(&prm));
    }
}

static inline size_t sizeof_tensor_type(int type)
{
    if (type == VX_TYPE_FLOAT32)
        return sizeof(vx_float32);;
    if (type == VX_TYPE_INT32)
        return sizeof(vx_int32);
    if (type == VX_TYPE_BOOL)
        return sizeof(vx_bool);

    return 0;
}

static inline size_t compute_patch_size(const size_t *view_end, size_t number_of_dimensions)
{
    size_t total_size = 1;
    for (size_t i = 0; i < number_of_dimensions; i++)
    {
        total_size *= view_end[i];
    }
    return total_size;
}

static inline void set_filenames(vx_char **filenames, vx_int32 input_num, vx_int32 num_params, vx_char *url)
{
    vx_int32 i = 0, j = 0;
    vx_int32 output_num = num_params - input_num;
    if (input_num > 1)
    {
        for (i = 0; i < input_num; i++)
        {
            snprintf(filenames[i], MAXPATHLENGTH, "%s/%s%d%s", url, "input", i + 1, ".dat");
        }
    }
    else
    {
        snprintf(filenames[i++], MAXPATHLENGTH, "%s/%s", url, "input.dat");
    }

    if (output_num > 1)
    {
        for (i = input_num, j = 0; i < num_params; i++, j++)
        {
            snprintf(filenames[i], MAXPATHLENGTH, "%s/%s%d%s", url, "output", j + 1, ".dat");
        }
    }
    else
    {
        snprintf(filenames[i], MAXPATHLENGTH, "%s/%s", url, "output.dat");
    }
}

static inline void query_tensor_parmeter(vx_kernel nn_kernel, vx_int32 *output_tensor_type, vx_size *output_num_dims, vx_size **output_dims, vx_int32 index)
{
    vx_int32 param_type;
    vx_status status = VX_SUCCESS;

    vx_parameter prm = vxGetKernelParameterByIndex(nn_kernel, index);

    if (NULL == prm)
    {
        status = VX_ERROR_INVALID_PARAMETERS;
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    }

    VX_CALL(vxQueryParameter(prm, VX_PARAMETER_TYPE, &param_type, sizeof(enum vx_type_e)));

    if (VX_TYPE_TENSOR == param_type)
    {
        vx_meta_format meta;

        VX_CALL(vxQueryParameter(prm, VX_PARAMETER_META_FORMAT, &meta, sizeof(vx_meta_format)));

        VX_CALL(vxQueryMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS,
                                           output_num_dims, sizeof(vx_size)));

        *output_dims = ct_alloc_mem(sizeof(vx_size) * (*output_num_dims));
        ASSERT(*output_dims);

        VX_CALL(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DIMS, *output_dims,
                                           sizeof(vx_size) * (*output_num_dims)));

        VX_CALL(vxQueryMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE,
                                           output_tensor_type, sizeof(vx_enum)));
    }

    VX_CALL(vxReleaseParameter(&prm));
}

static inline void calculate_tensor_strides(vx_size num_dims, vx_int32 tensor_type, vx_size *dims, vx_size *tensor_strides)
{
    vx_size j = 0;

    if (tensor_type == VX_TYPE_FLOAT32)
        *(tensor_strides) = sizeof(vx_float32);
    else if (tensor_type == VX_TYPE_INT32)
        *(tensor_strides) = sizeof(vx_int32);
    else if (tensor_type == VX_TYPE_BOOL)
        *(tensor_strides) = sizeof(vx_bool);

    for (j = 1; j < num_dims; j++)
    {
        tensor_strides[j] = tensor_strides[j - 1] * dims[j - 1];
    }
}

static inline enum vx_type_e tensor_vx_type(char * nnef_dtype)
{
    if (strcmp(nnef_dtype, "scalar") == 0)
    {
        return VX_TYPE_FLOAT32;
    }
    else if (strcmp(nnef_dtype, "integer") == 0)
    {
        return VX_TYPE_INT32;
    }
    else if (strcmp(nnef_dtype, "logical") == 0)
    {
        return VX_TYPE_BOOL;
    }
    return VX_TYPE_INVALID;
}

// query input parameters of kernel to create tensor objects and add to node
static inline void set_input_parameter_to_node(vx_context context, vx_node node, 
                                               vx_tensor tensors[MAX_TENSORS], vx_char **filenames, vx_int32 index)
{
    vx_int32 i = 0;
    vx_char fixed_point_precision = 0;
    vx_char perror[MAXLEN] = "";

    nnef_tensor_t input_tensor = nnef_new_tensor();

    vx_char *tensor_type_name;

    nnef_read_tensor(filenames[index], input_tensor, perror);

    // query NNEF tensor number of dim
    vx_size num_dims = nnef_get_tensor_rank(input_tensor);

    // alloc memory according to num_dims
    vx_size *dims = ct_alloc_mem(num_dims * sizeof(size_t));
    // query NNEF tensor number of dim
    nnef_get_tensor_dims(input_tensor, dims);

    // query tensor type
    tensor_type_name = nnef_get_tensor_dtype(input_tensor);
    vx_int32 tensor_type = tensor_vx_type(tensor_type_name);

    vx_size *view_start = ct_alloc_mem(num_dims * sizeof(vx_size));
    memset(view_start, 0, num_dims * sizeof(vx_size));
    vx_size *tensor_strides = ct_alloc_mem(sizeof(vx_size) * num_dims);
    ASSERT(tensor_strides && view_start);

    calculate_tensor_strides(num_dims, tensor_type, dims, tensor_strides);

    tensors[index] = vxCreateTensor(context, num_dims, dims, tensor_type, fixed_point_precision);
    ASSERT(tensors[index]);

    VX_CALL(vxCopyTensorPatch(tensors[index], num_dims, view_start, dims,
                              tensor_strides, (vx_char *)nnef_get_tensor_data(input_tensor), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    VX_CALL(vxSetParameterByIndex(node, index, (vx_reference)tensors[index]));

    ct_free_mem(dims);
    ct_free_mem(tensor_strides);
    ct_free_mem(view_start);
    nnef_free_tensor(input_tensor);
}

// query output parameters of kernel to create tensor objects and add to node
static inline void set_output_parameter_to_node(vx_context context, vx_node node, vx_kernel nn_kernel, vx_tensor tensors[MAX_TENSORS],
                                                vx_int32 *output_tensor_type, vx_size *output_num_dims, vx_size **output_dims,
                                                vx_size **output_tensor_strides, vx_size **output_view_start, vx_int32 index)
{
    vx_int32 i = 0;
    vx_char fixed_point_precision = 0;

    query_tensor_parmeter(nn_kernel, output_tensor_type, output_num_dims, output_dims, index);

    *output_view_start = ct_alloc_mem(*output_num_dims * sizeof(vx_size));
    memset(*output_view_start, 0, *output_num_dims * sizeof(vx_size));
    *output_tensor_strides = ct_alloc_mem(sizeof(vx_size) * (*output_num_dims));
    calculate_tensor_strides(*output_num_dims, *output_tensor_type, *output_dims, *output_tensor_strides);

    tensors[index] = vxCreateTensor(context, *output_num_dims, *output_dims,
                                    *output_tensor_type, fixed_point_precision);
    ASSERT(tensors[index]);

    VX_CALL(vxSetParameterByIndex(node, index, (vx_reference)tensors[index]));
}

static inline vx_float32 sqr(const vx_float32 x)
{
    return x * x;
}

//Calculate difference between NNEF kernel output and ref output dat 
static inline vx_float32 relative_difference(const vx_size n, const vx_float32 *ref, const vx_float32 *dat)
{
    vx_float32 diff = 0;
    vx_float32 range = 0;
    vx_size i = 0;
    for (i = 0; i < n; ++i)
    {
        diff += sqr(ref[i] - dat[i]);
        range += sqr(ref[i]);
    }
    return (vx_float32)sqrt(diff / range);
}

//Compare nnef output data with ground truth output dat files
static inline vx_status nnef_output_compare(vx_int32 *output_tensor_type, vx_size *output_num_dims, vx_char **output_data, 
                                            vx_int32 output_num, vx_char **filenames)
{
    vx_int32 i = 0;
    vx_status status = VX_SUCCESS;
    vx_char perror[MAXLEN] = "";
    nnef_tensor_t ground_truth_tensor = nnef_new_tensor();

    for (i = 0; i < output_num; i++)
    {
        nnef_read_tensor(filenames[i], ground_truth_tensor, perror);

        vx_size ground_truth_num_dims = nnef_get_tensor_rank(ground_truth_tensor);

        vx_size *ground_truth_dims = ct_alloc_mem(ground_truth_num_dims * sizeof(size_t));
        nnef_get_tensor_dims(ground_truth_tensor, ground_truth_dims);

        vx_int32 ground_truth_tensor_type = tensor_vx_type(nnef_get_tensor_dtype(ground_truth_tensor));

        vx_size ground_truth_volume = compute_patch_size(ground_truth_dims, ground_truth_num_dims);

        vx_char *ground_truth_data = (vx_char *)nnef_get_tensor_data(ground_truth_tensor);

        if (ground_truth_tensor_type != output_tensor_type[i])
        {
            printf("data-type of output_tensors does not match reference data-type \n");
            status = VX_FAILURE;
        }
        else if (ground_truth_num_dims != output_num_dims[i])
        {
            printf("shape of output_tensors does not match reference shape \n");
            status = VX_FAILURE;
        }
        else
        {
            if (ground_truth_tensor_type == VX_TYPE_FLOAT32) //NNEF tensor type is "scalar"
            {
                vx_float32 diff = relative_difference(ground_truth_volume,
                                                      (const vx_float32*)ground_truth_data,
                                                      (const vx_float32*)output_data[i]);
                if (diff > DIFF_EPSILON)
                {
                    printf("output diff %e\n", diff);
                    status = VX_FAILURE;  //The NNEF output difference is greater than 10e-4
                }
            }
            else
            {
                vx_size ground_truth_outputs_data_size = ground_truth_volume * sizeof_tensor_type(ground_truth_tensor_type);
                if (memcmp(ground_truth_data, output_data[i], ground_truth_outputs_data_size) != 0)
                {
                    printf("output does not match\n");
                    status = VX_FAILURE;  //The NNEF output does not match
                }
            }
        }
        ct_free_mem(ground_truth_dims);
    }

    nnef_free_tensor(ground_truth_tensor);

    return status;
}

TESTCASE(TensorNNEFImport, CT_VXContext, ct_setup_vx_context, 0)

TEST_WITH_ARG_DYNAMIC(TensorNNEFImport, testNNEFImport, nnef_import_arg)
{
    vx_status status = VX_SUCCESS;
    if (arg_ == NULL)
    {
        printf("\nFailed to find nnef data, check if you have nnef_core data in VX_TEST_DATA_PATH\n");
        status = VX_FAILURE;
        ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    }
    else
    {
        vx_context context = context_->vx_context_;
        vx_char *nnef_type = arg_->type;
        vx_char *url = arg_->url;

        vx_kernel nn_kernel = NULL;
        vx_node node = 0;
        vx_int32 i = 0;
        vx_size j = 0;
        
        vx_int32 input_num = 0, output_num = 0;
        vx_int32 num_params = 0;
        vx_char **filenames = NULL;
        vx_tensor vx_tensors[MAX_TENSORS] = { NULL };
        vx_char perror[MAXLEN] = "";

        //Get nnef kernel
        nn_kernel = vxImportKernelFromURL(context, nnef_type, url);

        VX_CALL(vxGetStatus((vx_reference)nn_kernel));

        // create OpenVX graph
        vx_graph graph = vxCreateGraph(context);

        node = vxCreateGenericNode(graph, nn_kernel);

        // query number of parameters in imported kernel
        vxQueryKernel(nn_kernel, VX_KERNEL_PARAMETERS, &num_params, sizeof(vx_uint32));

        // query input number of parameters in imported kernel
        query_input_output_num(nn_kernel, num_params, &input_num, &output_num);

        // memory alloc for tensors
        vx_int32 *output_tensor_type;
        vx_size *output_num_dims;
        vx_size **output_dims;
        vx_char **output_data;
        vx_size **output_tensor_strides;
        vx_size **output_view_start;

        output_tensor_type = mem_init(output_num * sizeof(vx_int32));
        output_num_dims = mem_init(output_num * sizeof(vx_size));
        output_dims = mem_init(output_num * sizeof(vx_size*));
        output_data = mem_init(output_num * sizeof(vx_char*));
        output_tensor_strides = mem_init(output_num * sizeof(vx_size*));
        output_view_start = mem_init(output_num * sizeof(vx_size*));

        // memory for filenames
        filenames = ct_alloc_mem(sizeof(vx_char *) * num_params);
        for (i = 0; i < num_params; i++)
        {
            filenames[i] = ct_alloc_mem(sizeof(vx_char) * MAXPATHLENGTH);
        }
        ASSERT(output_tensor_type && output_num_dims && output_dims && output_data && output_tensor_strides &&
               output_view_start && filenames);

        set_filenames(filenames, input_num, num_params, url);
        // query input parameters of kernel to create tensor objects and add to node
        for (i = 0; i < input_num; i++)
        {
            set_input_parameter_to_node(context, node, vx_tensors, filenames, i);

        }

        // query output parameters of kernel to create tensor objects and add to node
        for (i = input_num, j = 0; i < num_params; i++, j++)
        {
            set_output_parameter_to_node(context, node, nn_kernel, vx_tensors, &output_tensor_type[j], &output_num_dims[j], 
                                         &output_dims[j], &output_tensor_strides[j], &output_view_start[j], i);

            vx_size output_volume = compute_patch_size(output_dims[j], output_num_dims[j]);
            output_data[j] = ct_alloc_mem(*(output_tensor_strides[j]) * output_volume);
            ASSERT(output_data[j]);
        }

        // verify graph
        VX_CALL(vxVerifyGraph(graph));

        // execute the graph to run NNEF kernel
        VX_CALL(vxProcessGraph(graph));

        // reads out the output data into a C tensor data structure
        for (i = input_num, j = 0; i < num_params; i++, j++)
        {
            VX_CALL(vxCopyTensorPatch(vx_tensors[i], output_num_dims[j], output_view_start[j], output_dims[j],
                                      output_tensor_strides[j], output_data[j], VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
        }

        //Compare nnef output data with correct output dat file
        vx_char **output_filenames = filenames + input_num;
        VX_CALL(nnef_output_compare(output_tensor_type, output_num_dims, output_data, output_num, output_filenames));

        VX_CALL(vxReleaseNode(&node));

        for (i = 0; i < num_params; i++)
        {
            vxReleaseTensor(&vx_tensors[i]);
        }

        VX_CALL(vxReleaseKernel(&nn_kernel));

        VX_CALL(vxReleaseGraph(&graph));

        free_tensors_paramters(output_tensor_type, output_num_dims, output_dims,
                               output_data, output_tensor_strides, output_view_start,
                               filenames, num_params, output_num);

        ASSERT_EQ_VX_STATUS(VX_SUCCESS, status);
    }
}

TESTCASE_TESTS(TensorNNEFImport, testNNEFImport)

void CT_NNEFSetup()
{
    vx_int32 i = 0;
    char file_path[MAXPATHLENGTH];
    const vx_char *env = NULL;
    const vx_char *nnef_folder_name = "nnef_core";

    vx_char nnef_kernel_path[MAXPATHLENGTH];

    env = ct_get_test_file_path();

    env = ct_get_test_file_path();
    snprintf(file_path, MAXPATHLENGTH, "%s%s/", env, nnef_folder_name);

    file_num = CT_ListFolder(MAX_NNEF_KERNELS, file_path, file_names);

    if (file_num > 0)
    {
        for (i = 0; i < file_num; i++)
        {
            snprintf(nnef_kernel_path, MAXPATHLENGTH, "%s%s/%s", env, nnef_folder_name, file_names[i]);
            memcpy(nnef_kernel_url[i], nnef_kernel_path, sizeof(nnef_kernel_path));
#if defined(_WIN32)
            strcat(nnef_kernel_path, "*.*");
#endif
        }
        CT_TEST_WITH_ARG_SET(TensorNNEFImport, testNNEFImport, nnef_import_arg, file_names, nnef_kernel_url, file_num);
    }
}

void CT_NNEFTeardown()
{
    CT_TEST_WITH_ARG_FREE(TensorNNEFImport, testNNEFImport, nnef_import_arg, file_num);
}

#endif
