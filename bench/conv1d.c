// conv1d.c
/*
objdump -d -Mintel --disassemble=conv1d_avx2 conv1d
gcc conv1d.c -O3 -mavx2 -mfma -fopt-info-vec-optimized -lm -o conv1d
    > ./conv1d                                                            
    n = 1048576, ks = 5
    Scalar: 0.125 ms, 84.14 GFLOP/s
    AVX2  : 0.210 ms, 49.92 GFLOP/s
    Max abs diff = 0
    i=0  scalar=0.028837  avx2=0.028837
    i=1  scalar=0.069964  avx2=0.069964
    i=2  scalar=0.131490  avx2=0.131490
    i=3  scalar=0.196748  avx2=0.196748
    i=4  scalar=0.261422  avx2=0.261422

gcc conv1d.c -O3 -fno-tree-vectorize -mavx2 -mfma -lm -o conv1d_novec
    > ./conv1d_novec
    n = 1048576, ks = 5
    Scalar: 0.578 ms, 18.15 GFLOP/s
    AVX2  : 0.204 ms, 51.45 GFLOP/s
    Max abs diff = 0
    i=0  scalar=0.028837  avx2=0.028837
    i=1  scalar=0.069964  avx2=0.069964
    i=2  scalar=0.131490  avx2=0.131490
    i=3  scalar=0.196748  avx2=0.196748
    i=4  scalar=0.261422  avx2=0.261422

*/
#include <stddef.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define KS 5

void conv1d_scalar(float *y, const float *x, const float *kernel,
                   int n, int ks) {
    int ks2 = ks / 2;
    const float *x_center = x + ks2; 

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = -ks2; k <= ks2; k++) {
            float xv = x_center[i - k];
            float kv = kernel[k + ks2];
            sum += kv * xv;
        }
        y[i] = sum;
    }
}

void conv1d_avx2(float * __restrict y, const float * __restrict x, const float * __restrict kernel, int n, int ks) {
    const int ks2 = ks / 2;          // = 2
    const float *x_center = x + ks2;

    __m256 ktap[5];
    for (int t = 0; t < 5; ++t) {
        ktap[t] = _mm256_set1_ps(kernel[t]);
    }

    for (int i = 0; i < n; i += 8) {
        const float *xi = x_center + i;
        __m256 sum = _mm256_setzero_ps();

        static const int off[5] = { 2, 1, 0, -1, -2 };

        for (int t = 0; t < 5; ++t) {
            __m256 xvec = _mm256_loadu_ps(xi + off[t]);
            sum = _mm256_fmadd_ps(xvec, ktap[t], sum);
        }

        _mm256_storeu_ps(&y[i], sum);
    }
}

// ----------------------
// Helpers
// ----------------------

static double seconds_since(const struct timespec *start, const struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) * 1e-9;
}

// Reflective padding like the book: xp has length n + 2*ks2
static void pad_signal_reflect(const float *x, int n, float *xp, int ks) {
    int ks2 = ks / 2;
    int n2 = n + 2 * ks2;

    // center portion: copy original signal
    for (int i = 0; i < n; i++) { 
        xp[ks2 + i] = x[i];
    }

    // left pad: reflect
    for (int i = 0; i < ks2; i++) {
        xp[i] = x[ks2 - i - 1];              // mirror of first ks2 samples
    }

    // right pad: reflect
    for (int i = 0; i < ks2; i++) {
        xp[ks2 + n + i] = x[n - i - 1];      // mirror of last ks2 samples
    }

    (void)n2; // just to document the relationship; not used further
}

// ----------------------
// Main test
// ----------------------

int main(void) {
    const int n = 1 << 20;  // 1,048,576 samples
    int ks = KS;
    const int ks2 = ks / 2;
    const int n_padded = n + 2 * ks2;

    // Sanity checks so conv1d_avx2 can assume a pure vector path.
    if (n % 8 != 0) {
        fprintf(stderr, "Error: n (%d) must be a multiple of 8 for conv1d_avx2.\n", n);
        return 1;
    }
    if (ks != 5) {
        fprintf(stderr, "Error: ks (%d) must be 5 for the specialized AVX2 kernel.\n", ks);
        return 1;
    }

    // Simple Gaussian-ish kernel like in the book
    float kernel[KS] = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

    // Allocate arrays
    float *x      = (float *)aligned_alloc(32, n * sizeof(float));
    float *xp     = (float *)aligned_alloc(32, n_padded * sizeof(float));
    float *y_ref  = (float *)aligned_alloc(32, n * sizeof(float));
    float *y_avx  = (float *)aligned_alloc(32, n * sizeof(float));

    if (!x || !xp || !y_ref || !y_avx) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Initialize input with some deterministic “signal”
    for (int i = 0; i < n; i++) {
        float t = (float)i * 0.001f;
        float signal = sinf(2.0f * 3.14159265f * 5.0f * t)
                     + 0.5f * sinf(2.0f * 3.14159265f * 11.0f * t);
        x[i] = signal;
    }

    // Pad the signal
    pad_signal_reflect(x, n, xp, ks);

    // Warmup runs
    conv1d_scalar(y_ref, xp, kernel, n, ks);
    conv1d_avx2(y_avx, xp, kernel, n, ks);

    // Time scalar
    struct timespec t0, t1;
    int reps = 5;

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) {
        conv1d_scalar(y_ref, xp, kernel, n, ks);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_scalar = seconds_since(&t0, &t1) / reps;

    // Time AVX2
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) {
        conv1d_avx2(y_avx, xp, kernel, n, ks);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt_avx = seconds_since(&t0, &t1) / reps;

    // Compare outputs
    float max_abs_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(y_ref[i] - y_avx[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    // FLOP counting: each output does ks multiply-adds = 2*ks FLOPs
    double flops = 2.0 * (double)ks * (double)n;
    double gflops_scalar = flops / dt_scalar / 1e9;
    double gflops_avx    = flops / dt_avx / 1e9;

    printf("n = %d, ks = %d\n", n, ks);
    printf("Scalar: %.3f ms, %.2f GFLOP/s\n",
           dt_scalar * 1e3, gflops_scalar);
    printf("AVX2  : %.3f ms, %.2f GFLOP/s\n",
           dt_avx * 1e3, gflops_avx);
    printf("Max abs diff = %g\n", max_abs_diff);

    for (int i = 0; i < 5; i++) {
        printf("i=%d  scalar=%f  avx2=%f\n", i, y_ref[i], y_avx[i]);
    }

    free(x);
    free(xp);
    free(y_ref);
    free(y_avx);
    return 0;
}

