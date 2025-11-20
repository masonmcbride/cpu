/*
 * gcc dgemm.c -o dgemm \
  -I${MKLROOT}/include \
  -L${MKLROOT}/lib/intel64 \
  -lmkl_rt -lpthread -lm -ldl -O3 -march=native -ffast-math
 */
#include <stdio.h>
#include <stdlib.h>
#include <mkl_cblas.h>
#include <time.h>

int main() {
    int n = 1024;
    double *a = calloc(n * n, sizeof(double));
    double *b = calloc(n * n, sizeof(double));
    double *c = calloc(n * n, sizeof(double));

    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        a[i] = b[i] = 1.0;
    }

    // Warm-up MKL (not timed)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, a, n, b, n, 0.0, c, n);

    // Accurate high-resolution timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Timed DGEMM
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, a, n, b, n, 0.0, c, n);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = end.tv_sec - start.tv_sec +
                     1e-9 * (end.tv_nsec - start.tv_nsec);

    printf("MKL dgemm took %.6f seconds\n", elapsed);

    free(a); free(b); free(c);
    return 0;
}

