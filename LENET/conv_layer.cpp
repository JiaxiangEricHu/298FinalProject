#include <ap_int.h>
#include <ap_fixed.h>
#include "hls_stream.h"

#define IN_CH       1
#define OUT_CH      6
#define IN_H        32
#define IN_W        32
#define KERNEL      5
#define STRIDE      1
#define POOL_STRIDE 2
#define OUT_H_CONV  ((IN_H - KERNEL)/STRIDE + 1)
#define OUT_W_CONV  ((IN_W - KERNEL)/STRIDE + 1)
#define OUT_H_POOL  (OUT_H_CONV/POOL_STRIDE)
#define OUT_W_POOL  (OUT_W_CONV/POOL_STRIDE)

// 输入的二维的尺寸IN_H × IN_W；
// Convolution：输入的通道数IN_CH，输出的通道数OUT_CH；kernel_size是KERNEL；步长Stride是STRIDE
// Pooling：POOL_STRIDE是尺寸与步长

// 数据类型量化示例
typedef ap_fixed<16,8> data_t;
typedef data_t weight_t;
typedef data_t bias_t;

void conv_relu_pool(
    hls::stream< data_t > & in_stream,
    hls::stream< data_t > & out_stream,
    weight_t weights[OUT_CH][IN_CH][KERNEL][KERNEL],
    bias_t   bias   [OUT_CH]
    )
{
    #pragma HLS INTERFACE axis                                    port=in_stream
    #pragma HLS INTERFACE axis                                    port=out_stream
    #pragma HLS INTERFACE m_axi depth= OUT_CH*IN_CH*KERNEL*KERNEL port=weights    offset=slave   bundle=BUS_WEIGHTS
    #pragma HLS INTERFACE m_axi depth= OUT_CH                     port=bias       offset=slave   bundle=BUS_BIAS
    #pragma HLS INTERFACE s_axilite                               port=return                    bundle=CTRL

    static data_t input_buf[IN_CH][IN_H][IN_W];
    #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1

    static data_t conv_buf[OUT_CH][OUT_H_CONV][OUT_W_CONV];
    #pragma HLS ARRAY_PARTITION variable=conv_buf complete dim=1

    static data_t pool_buf[OUT_CH][OUT_H_POOL][OUT_W_POOL];
    #pragma HLS ARRAY_PARTITION variable=pool_buf complete dim=1

    for(int c=0; c<IN_CH; c++){
      for(int i=0; i<IN_H; i++){
        for(int j=0; j<IN_W; j++){
          #pragma HLS PIPELINE II=1
          data_t v = in_stream.read();
          input_buf[c][i][j] = v;
        }
      }
    }

    for(int co=0; co<OUT_CH; co++){
      for(int h=0; h<OUT_H_CONV; h++){
        for(int w=0; w<OUT_W_CONV; w++){
          #pragma HLS PIPELINE II=1
          data_t sum = bias[co];
          for(int ci=0; ci<IN_CH; ci++){
            for(int kh=0; kh<KERNEL; kh++){
              for(int kw=0; kw<KERNEL; kw++){
                #pragma HLS UNROLL factor=KERNEL
                sum += weights[co][ci][kh][kw] * input_buf[ci][h*STRIDE + kh][w*STRIDE + kw];
              }
            }
          }

          data_t activated = (sum > 0) ? sum : (data_t)0;
          conv_buf[co][h][w] = activated;
        }
      }
    }

  for(int co=0; co<OUT_CH; co++){
      for(int h=0; h<OUT_H_POOL; h++){
        for(int w=0; w<OUT_W_POOL; w++){
          #pragma HLS PIPELINE II=1
          data_t maxv = input_buf[0][0][0]; // 初始化一个值
          maxv = conv_buf[co][h*POOL_STRIDE][w*POOL_STRIDE];
          for(int ph=0; ph<POOL_STRIDE; ph++){
            for(int pw=0; pw<POOL_STRIDE; pw++){
              data_t val = conv_buf[co][h*POOL_STRIDE + ph][w*POOL_STRIDE + pw];
              if(val > maxv) maxv = val;
            }
          }
          pool_buf[co][h][w] = maxv;
        }
      }
    }

    for(int co=0; co<OUT_CH; co++){
      for(int i=0; i<OUT_H_POOL; i++){
        for(int j=0; j<OUT_W_POOL; j++){
          #pragma HLS PIPELINE II=1
          out_stream.write(pool_buf[co][i][j]);
        }
      }
    }
}
