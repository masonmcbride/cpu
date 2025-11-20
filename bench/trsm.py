import os
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"

import numpy as np
from scipy.linalg.blas import dtrsm
import time

n = 4096

# A will be a well-conditioned lower triangular matrix
A = np.tril(np.random.rand(n, n))
# Ensure A is not singular
A += np.eye(n) * 1.0

# B is general dense
B = np.random.rand(n, n)

# Warm-up
_ = dtrsm(1.0, A, B, side=1, lower=1, trans_a=0, diag=0, overwrite_b=0)

# Time it
start = time.perf_counter()
_ = dtrsm(1.0, A, B, side=1, lower=1, trans_a=0, diag=0, overwrite_b=0)
end = time.perf_counter()

# TRSM flops: roughly n² * n / 2 (since triangular)
gflops = (n**3 / 2) / (end - start) / 1e9

print(f"Single-threaded dtrsm: {end - start:.6f} s = {gflops:.2f} GFLOPs")
