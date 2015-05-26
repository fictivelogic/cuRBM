#include <cassert>
#include <math.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"

#define NUM_MOVIES 17770
#define NUM_USERS 458293


float sigmoid (float input); //TODO: Log probability?

__global__
void createInitialHidden(const int * const train_points,
						 const int * user_start,
						 const int * user_length,
						 const int * const b_hid,
						 float * intial_hiddens,
						 int num_hidden,
						 float * output) {

  // TODO: do not modify code, just comment on suboptimal accesses
	//

	int i_start = user_start[threadIdx.x + (blockDim.x * threadIdx.y)];
	int i_length = user_length[threadIdx.x + (blockDim.x * threadIdx.y)];
	//Need to know what size the thread block is so we can determine how many users each thread should attend to

	for (int i = 0; i < num_hidden < i++) //100 loops
	{
		float dot_prod = 0;
		for (int m = 0; m < i_length; m++)
		{
			int rating = train_points[i_start + (m * 2)];
			dot_prod += W[movie][rating][i];
			intial_hiddens[i] = sigmoid(dot_prod - b_hid[i]);
		}
	}
}