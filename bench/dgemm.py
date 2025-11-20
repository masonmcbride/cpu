import numpy as np
import time

# Size of the matrices
n = 4096

# Allocate and initialize matrices
A = np.ones((n, n), dtype=np.float64)
B = np.ones((n, n), dtype=np.float64)

# Warm-up (optional, ensures cache/warming JIT paths)
_ = A @ B

# Timing only the multiplication
start = time.perf_counter()
C = A @ B
end = time.perf_counter()

print(f"NumPy dgemm took {end - start:.6f} seconds")

