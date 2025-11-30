#include "Lenet.h"
#include <hls_math.h>

#include <cstring>
inline int idx3(int c, int h, int w, int H, int W) { return (c*H + h)*W + w; }

// Simple 2D convolution for single input channel block; supports multi-in multi-out by loops
static void conv2d(
    const data_t *in, int in_ch, int in_h, int in_w,
    const data_t *weights, // weights layout: out_ch, in_ch, k, k
    const data_t *biases,
    data_t *out, int out_ch, int k)
{
#pragma HLS INLINE off
// For every output channel
    for (int oc = 0; oc < out_ch; oc++) {
        for (int oh = 0; oh <= in_h - k; oh++) {
            for (int ow = 0; ow <= in_w - k; ow++) {
            #pragma HLS PIPELINE II=1
            data_t acc = biases[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int kh = 0; kh < k; kh++) {
                    for (int kw = 0; kw < k; kw++) {
                        int in_index = idx3(ic, oh+kh, ow+kw, in_h, in_w);
                        int w_index = ((oc*in_ch + ic)*k + kh)*k + kw;
                        acc += in[in_index] * weights[w_index];
                        }
                    }
                }
            int out_index = idx3(oc, oh, ow, in_h - k + 1, in_w - k + 1);
            out[out_index] = acc;
            }
        }
    }
}

static void relu_inplace(data_t *buf, int size) {
    for (int i = 0; i < size; i++) {
    #pragma HLS PIPELINE II=1
    data_t v = buf[i];
    buf[i] = (v > 0) ? v : 0;
    }
}

static void maxpool2x2(const data_t *in, int ch, int in_h, int in_w, data_t *out) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    for (int c = 0; c < ch; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
            #pragma HLS PIPELINE II=1
            data_t a = in[idx3(c, h*2+0, w*2+0, in_h, in_w)];
            data_t b = in[idx3(c, h*2+0, w*2+1, in_h, in_w)];
            data_t c0 = in[idx3(c, h*2+1, w*2+0, in_h, in_w)];
            data_t d = in[idx3(c, h*2+1, w*2+1, in_h, in_w)];
            data_t m = a;
            if (b > m) m = b;
            if (c0 > m) m = c0;
            if (d > m) m = d;
            out[idx3(c, h, w, out_h, out_w)] = m;
            }
        }
    }
}

static void fc_layer(const data_t *in, int in_len, const data_t *weights, const data_t *biases, data_t *out, int out_len) {
    for (int o = 0; o < out_len; o++) {
        data_t acc = biases[o];
        for (int i = 0; i < in_len; i++) {
            #pragma HLS PIPELINE II=1
            acc += weights[o*in_len + i] * in[i];
        }
    out[o] = acc;
    }
}

static void softmax_inplace(data_t *v, int n) {
    data_t maxv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > maxv) maxv = v[i];
        data_t sum = 0;
        for (int i = 0; i < n; i++) {
            v[i] = hls::expf(v[i] - maxv);
            sum += v[i];
        }
        for (int i = 0; i < n; i++) v[i] /= sum;
}

extern "C" void lenet_accel(
const data_t *images,
const data_t *params,
data_t *outputs,
int num_images
) {
#pragma HLS INTERFACE m_axi port=images offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=outputs offset=slave bundle=gmem2


#pragma HLS INTERFACE s_axilite port=images bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=outputs bundle=control
#pragma HLS INTERFACE s_axilite port=num_images bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

int off = 0;
const data_t *conv1_w = params + off; off += C1_OUT * IMG_CH * C1_K * C1_K;
const data_t *conv1_b = params + off; off += C1_OUT;
const data_t *conv2_w = params + off; off += C2_OUT * C1_OUT * C2_K * C2_K;
const data_t *conv2_b = params + off; off += C2_OUT;
const data_t *fc1_w = params + off; off += FC1_OUT * FC1_IN;
const data_t *fc1_b = params + off; off += FC1_OUT;
const data_t *fc2_w = params + off; off += FC2_OUT * FC1_OUT;
const data_t *fc2_b = params + off; off += FC2_OUT;
const data_t *fc3_w = params + off; off += FC3_OUT * FC2_OUT;
const data_t *fc3_b = params + off; off += FC3_OUT;

// Buffers in PL (on-stack arrays)
static data_t buf1[C1_OUT * (IMG_H - C1_K + 1) * (IMG_W - C1_K + 1)]; // conv1 output: 6x24x24
static data_t buf1p[C1_OUT * P1_H * P1_W]; // pool1: 6x12x12
static data_t buf2[C2_OUT * (P1_H - C2_K + 1) * (P1_W - C2_K + 1)]; // conv2 out: 16x8x8
static data_t buf2p[C2_OUT * P2_H * P2_W]; // pool2: 16x4x4
static data_t fc1_in[FC1_IN]; // 256
static data_t fc1_out[FC1_OUT];
static data_t fc2_out[FC2_OUT];
static data_t logits[FC3_OUT];

#pragma HLS BIND_STORAGE variable=buf1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buf1p type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buf2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=buf2p type=ram_2p impl=bram

#pragma HLS ARRAY_PARTITION variable=buf1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=buf1p cyclic factor=4 dim=1

#pragma HLS BIND_STORAGE variable=fc1_in type=ram_1p impl=bram


// Process each image
for (int n = 0; n < num_images; n++) {
// Read image into a small input buffer
    static data_t in_img[IMG_CH * IMG_H * IMG_W];
    #pragma HLS ARRAY_PARTITION variable=in_img cyclic factor=8 dim=1
    int img_offset = n * IMG_CH * IMG_H * IMG_W;

    for (int i = 0; i < IMG_CH*IMG_H*IMG_W; i++) {
    #pragma HLS PIPELINE II=1
    in_img[i] = images[img_offset + i];
    }


    // conv1: in 1x28x28 -> out 6x24x24
    conv2d(in_img, IMG_CH, IMG_H, IMG_W, conv1_w, conv1_b, buf1, C1_OUT, C1_K);
    relu_inplace(buf1, C1_OUT * (IMG_H - C1_K + 1) * (IMG_W - C1_K + 1));
    // pool1: 6x24x24 -> 6x12x12
    maxpool2x2(buf1, C1_OUT, IMG_H - C1_K + 1, IMG_W - C1_K + 1, buf1p);


    // conv2: in 6x12x12 -> out 16x8x8
    conv2d(buf1p, C1_OUT, P1_H, P1_W, conv2_w, conv2_b, buf2, C2_OUT, C2_K);
    relu_inplace(buf2, C2_OUT * (P1_H - C2_K + 1) * (P1_W - C2_K + 1));
    // pool2: 16x8x8 -> 16x4x4
    maxpool2x2(buf2, C2_OUT, P1_H - C2_K + 1, P1_W - C2_K + 1, buf2p);


    // flatten to fc1_in
    for (int i = 0; i < FC1_IN; i++) {
        #pragma HLS PIPELINE II=1
        fc1_in[i] = buf2p[i];
    }


    // fc1: 256 -> 120
    fc_layer(fc1_in, FC1_IN, fc1_w, fc1_b, fc1_out, FC1_OUT);
    relu_inplace(fc1_out, FC1_OUT);


    // fc2: 120 -> 84
    fc_layer(fc1_out, FC1_OUT, fc2_w, fc2_b, fc2_out, FC2_OUT);
    relu_inplace(fc2_out, FC2_OUT);


    // fc3: 84 -> 10
    fc_layer(fc2_out, FC2_OUT, fc3_w, fc3_b, logits, FC3_OUT);


    // softmax
    softmax_inplace(logits, FC3_OUT);


    // write output (probabilities)
    int out_offset = n * FC3_OUT;
    for (int i = 0; i < FC3_OUT; i++) {
        #pragma HLS PIPELINE II=1
        outputs[out_offset + i] = logits[i];
        }
    }
}
