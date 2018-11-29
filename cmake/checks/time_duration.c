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
#include <stdint.h>

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h> // gettimeofday
#endif
#include <time.h>
#if defined __MACH__ && defined __APPLE__
#include <mach/mach_time.h>
#endif

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h> // QueryPerformanceFrequency / QueryPerformanceCounter
#endif

static int64_t CT_getTickCount(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (int64_t)counter.QuadPart;
#elif defined __linux || defined __linux__
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (int64_t)tp.tv_sec * 1e9 + tp.tv_nsec;
#elif defined __MACH__ && defined __APPLE__
    return (int64_t)mach_absolute_time();
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (int64_t)tv.tv_sec * 1e6 + tv.tv_usec;
#endif
}

static double CT_getTickFrequency(void)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double)freq.QuadPart;
#elif defined __linux || defined __linux__
    return 1e9;
#elif defined __MACH__ && defined __APPLE__
    static double freq = 0;
    if(freq == 0)
    {
        mach_timebase_info_data_t sTimebaseInfo;
        mach_timebase_info(&sTimebaseInfo);
        freq = sTimebaseInfo.denom * 1e9 / sTimebaseInfo.numer;
    }
    return freq;
#else
    return 1e6;
#endif
}

int main()
{
    double f = CT_getTickFrequency();
    int64_t v = CT_getTickCount();
    return (f > 0. && v > 0) ? 0 : 1; // result is not used
}
