//
// Created by samuel on 5/24/15.
//

#include <cuda_runtime.h>
#include "createMovieRatings.cuh"


__global__ void
createMovieRatingsKernel(const float *weights, const float *hidden_features,
    float* movie_rating_probs, int num_movies, int num_hidden_features) {

    // weights[NUM_MOVIES][5][NUM_FEATURES]
    //
    // movie_rating_index = movie_id * 5 + rating_id
    //      (index of current movie_id/rating_id pair)
    unsigned int movie_rating_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = 0;
    float dot_prod; // temporary, local dot product variable
    while (movie_rating_index < num_movies * 5) {
        dot_prod = 0.00; // Initialize the dot product to 0

        for (i = 0; i < num_hiddens; i++) {
            // Indexing: weights[movie_id][rating_id][feature_id]
            // movie_id - [1, 17771]
            // rating_id - [0, 4]
            // feature_id - [0, 99]
            dot_prod += weights[movie_rating_index*num_hidden_features + i]
                        * hidden_features[i]; // Do the dot product
        }
        // store the dot_product result
        movie_rating_probs[movie_rating_index] = dot_prod;

        // re-use this thread on another data point:
        movie_rating_index += blockDim.x * gridDim.x;
    }
}
