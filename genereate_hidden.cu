#include <cassert>
#include <math.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"

#define NUM_MOVIES 17770
#define NUM_USERS 458293

__global__
void createInitialHidden(const int * const train_points,
						 const int * user_start,
						 const int * user_length,
						 int num_hidden,
						 float * output) {

  // TODO: do not modify code, just comment on suboptimal accesses

	int i_start = user_start(threadIdx.x + (blockDim.x * threadIdx.y));


	const int i = threadIdx.x + 64 * blockIdx.x;
	int j = 4 * threadIdx.y + 64 * blockIdx.y;
	const int end_j = j + 4;

	for (; j < end_j; j++) {
	  output[j + n * i] = input[i + n * j];
	}
}