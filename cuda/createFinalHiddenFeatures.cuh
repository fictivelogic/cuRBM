#ifndef NETFLIX_RBM_CREATEFINALHIDDENFEATURES_H
#define NETFLIX_RBM_CREATEFINALHIDDENFEATURES_H

// Third step of RBM
// Get final hidden feature probabilities from movie rating probabilities
__global__ void createFinalHiddenUnitsKernel(const float *weights,
                                             const float *movie_rating_probs,
                                             float* final_hidden_features,
                                             int num_movies,
                                             int num_hidden_features);

#endif //NETFLIX_RBM_CREATEFINALHIDDENFEATURES_H
