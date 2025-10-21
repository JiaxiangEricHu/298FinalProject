#include <stdio.h>

#define N 16

void matmul(int a[N][N], int b[N][N], int c[N][N]) {
  int buffer_a[N][N];
  int buffer_b[N][N];
  int buffer_c[N][N];

#pragma HLS ARRAY_PARTITION cyclic variable = buffer_a factor = N
#pragma HLS ARRAY_PARTITION cyclic variable = buffer_b factor = N
#pragma HLS ARRAY_PARTITION cyclic variable = buffer_c factor = N

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
      buffer_a[i][j] = a[i][j];
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
      buffer_b[i][j] = b[i][j];
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
      buffer_c[i][j] = c[i][j];
    }
  }

loop1:
  for (int i = 0; i < N; i++) {
  loop2:
    for (int j = 0; j < N; j++) {
    loop3:
      for (int k = 0; k < N; k++) {
#pragma HLS UNROLL factor = N
        buffer_c[i][j] += buffer_a[i][k] * buffer_b[k][j];
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE
      c[i][j] = buffer_c[i][j];
    }
  }
}

int main() {
  int a[N][N];
  int b[N][N];
  int c[N][N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] = 4;
      b[i][j] = 2;
      c[i][j] = 0;
    }
  }

  matmul(a, b, c);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d ", c[i][j]);
    }
    printf("\n");
  }
}
