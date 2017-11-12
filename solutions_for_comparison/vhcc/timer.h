
#pragma once

// A simple timer class

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

class cputimer
{
	timespec start;
	timespec end;
public:
	void start_timer()
	{
		int res = clock_gettime(CLOCK_MONOTONIC, &start);
	}
	float milliseconds_elapsed()
	{
		int res = clock_gettime(CLOCK_MONOTONIC, &end);

		timespec temp;
		if (end.tv_nsec - start.tv_nsec < 0) {
			temp.tv_sec = end.tv_sec - start.tv_sec - 1;
			temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec - start.tv_sec;
			temp.tv_nsec = end.tv_nsec - start.tv_nsec;
		}
		return temp.tv_sec * 1000.0 + ((double)temp.tv_nsec/1000000.0);
	}
	float nanoseconds_elapsed()
	{
		int res = clock_gettime(CLOCK_MONOTONIC, &end);

		timespec temp;
		if (end.tv_nsec - start.tv_nsec < 0) {
			temp.tv_sec = end.tv_sec - start.tv_sec - 1;
			temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec - start.tv_sec;
			temp.tv_nsec = end.tv_nsec - start.tv_nsec;
		}
		return temp.tv_sec * 1000000000.0 + ((double)temp.tv_nsec);
	}
};



/*  Adapted:
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifdef __CUDACC__

#include <cuda.h>
#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#include "cuda.h"

class gputimer
{
    cudaEvent_t start;
    cudaEvent_t end;

    public:
    timer()
    { 
        // DO Nothing!
    }

    /*  New method to start timer */
    void start_timer()
    {
        CUDA_SAFE_CALL(cudaEventCreate(&start)); 
        CUDA_SAFE_CALL(cudaEventCreate(&end));
        CUDA_SAFE_CALL(cudaEventRecord(start,0));
    }

    /* New method added for stopping timer so that we can use the same timer */
    void stop_timer()
    {
        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(end));
    }

    ~timer()
    {
        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(end));
    }

    float milliseconds_elapsed()
    { 
        float elapsed_time;
        CUDA_SAFE_CALL(cudaEventRecord(end, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
        return elapsed_time;
    }
    float seconds_elapsed()
    { 
        return milliseconds_elapsed() / 1000.0;
    }
};

#endif
