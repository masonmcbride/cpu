/*
gcc trsm.c -o trsm \
  -I${MKLROOT}/include \
  -L${MKLROOT}/lib/intel64 \
  -lmkl_rt -lpthread -lm -ldl -O3 -march=native
*/
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <time.h>

int main() {
    int n = 4096;

    // Allocate column-major arrays (BLAS expects column-major)
    double *a = calloc(n * n, sizeof(double));
    double *b = calloc(n * n, sizeof(double));

    // Initialize A as a triangular matrix (lower) and B as RHS
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {  // lower triangle
            a[j * n + i] = drand48();
        }
    }
    for (int i = 0; i < n * n; i++) {
        b[i] = drand48();
    }

    double alpha = 1.0;

    // Warm-up
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                n, n, alpha, a, n, b, n);

    // Time it
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                n, n, alpha, a, n, b, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec + 1e-9 * (end.tv_nsec - start.tv_nsec);

    double flops = 1.0 * n * n * n; // TRSM is ~n^3 ops
    printf("MKL trsm (n=%d) took %.6f seconds = %.2f GFLOPs\n",
           n, elapsed, flops / elapsed / 1e9);

    free(a);
    free(b);
    return 0;
}
