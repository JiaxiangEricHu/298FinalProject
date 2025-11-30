#ifndef LENET_HLS_H
#define LENET_HLS_H


#include <ap_int.h>
#include <hls_stream.h>
#include <ap_fixed.h>


// Using float for clarity; replace with ap_fixed if you want fixed-point.
typedef float data_t;


// LeNet dimensions (LeNet-5 classic)
#define IMG_CH 1
#define IMG_H 28
#define IMG_W 28


#define C1_OUT 6
#define C1_K 5


#define P1_H 12
#define P1_W 12


#define C2_OUT 16
#define C2_K 5


#define P2_H 4
#define P2_W 4


#define FC1_IN (C2_OUT * P2_H * P2_W) // 16 * 4 * 4 = 256
#define FC1_OUT 120
#define FC2_OUT 84
#define FC3_OUT 10


// Top-level HLS entry
extern "C" {
void lenet_accel(
const data_t *images, // pointer to input images (num_images x 28 x 28)
const data_t *params, // pointer to packed params (weights + biases)
data_t *outputs, // pointer to outputs (num_images x 10)
int num_images // number of images to process
);
}
#endif // LENET_HLS_H
