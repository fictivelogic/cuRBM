#ifndef NETFLIX_RBM_CREATEMOVIERATINGS_H
#define NETFLIX_RBM_CREATEMOVIERATINGS_H

// Second step of RBM
// Get movie rating probabilities from initial hidden feature probabilities
__global__ void createMovieRatingsKernel(
                    const float *weights,
                    const float *initial_hidden_features,
                    float* movie_rating_probs,
                    int num_movies,
                    int num_hidden_features);

#endif //NETFLIX_RBM_CREATEMOVIERATINGS_H
