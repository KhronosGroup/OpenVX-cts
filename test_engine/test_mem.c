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

#include <stdlib.h>
#include <string.h>
#include <VX/vx.h>

void *ct_alloc_mem(size_t size)
{
    void *ptr = NULL;

    if (0 != size)
    {
        ptr = malloc(size);
    }

    return (ptr);
}

void ct_free_mem(void *ptr)
{
    if (NULL != ptr)
    {
        free(ptr);
    }
}

void ct_memset(void *ptr, vx_uint8 c, size_t size)
{
    if (NULL != ptr)
    {
        memset(ptr, c, size);
    }
}

void *ct_calloc(size_t nmemb, size_t size)
{
    void *ptr = NULL;

    if ((0 != nmemb) && (0 != size))
    {
        ptr = calloc(nmemb, size);
    }

    return (ptr);
}
