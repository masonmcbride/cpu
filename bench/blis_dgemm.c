/*
  gcc blis_dgemm.c -o blis_dgemm \
  -I$HOME/w/cs/fun/blis/include/haswell \
  -L$HOME/w/cs/fun/blis/lib/haswell \
  -Wl,-rpath,$HOME/w/cs/fun/blis/lib/haswell \
  -lblis -fopenmp -lm -O3 -march=native
*/
#include <stdio.h>
#include <stdlib.h>
#include "blis.h"
#include <time.h>

int main() {
    dim_t n = 1024;
    obj_t a, b, c;
    obj_t alpha, beta;

    bli_init();

    bli_thread_set_num_threads( 20 );

    // Create scalar alpha and beta
    bli_obj_scalar_init_detached(BLIS_DOUBLE, &alpha);
    bli_obj_scalar_init_detached(BLIS_DOUBLE, &beta);
    bli_setsc(1.0, 0.0, &alpha); // real=1.0, imag=0.0
    bli_setsc(0.0, 0.0, &beta);

    // Create BLIS matrices
    bli_obj_create(BLIS_DOUBLE, n, n, 0, 0, &a);
    bli_obj_create(BLIS_DOUBLE, n, n, 0, 0, &b);
    bli_obj_create(BLIS_DOUBLE, n, n, 0, 0, &c);

    // Initialize matrices
    bli_randm(&a);
    bli_randm(&b);
    bli_setm(&BLIS_ZERO, &c);

    // Warm-up
    bli_gemm(&alpha, &a, &b, &beta, &c);
    printf("A row stride: %ld\n", bli_obj_row_stride(&a));
    printf("A col stride: %ld\n", bli_obj_col_stride(&a));

    // Timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    bli_gemm(&alpha, &a, &b, &beta, &c);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);

    double flops = 2.0 * n * n * n;
    printf("BLIS gemm (n=%lu) took %.6f seconds = %.2f GFLOPs\n",
           n, elapsed, flops / elapsed / 1e9);

    // Cleanup
    bli_obj_free(&a);
    bli_obj_free(&b);
    bli_obj_free(&c);
    bli_finalize();

    return 0;
}
