//
// Created by samuel on 5/24/15.
//

#ifndef NETFLIX_RBM_CREATEMOVIERATINGS_H
#define NETFLIX_RBM_CREATEMOVIERATINGS_H


// second loop of the pseudo-code
__global__ void createMovieRatingsKernel(const float *weights,
                                         const float *hidden_features,
                                         float* movie_rating_probs,
                                         int num_movies,
                                         int num_hidden_features);

#endif //NETFLIX_RBM_CREATEMOVIERATINGS_H
