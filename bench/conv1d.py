# conv1d.py
import time 
import numpy as np
from tinygrad import Tensor, Context, Device, GlobalCounters, dtypes, nn
from tinygrad.uop.ops import UOp, AxisType, KernelInfo
from tinygrad.dtype import AddrSpace
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.helpers import getenv

# simple 1D conv:
#   y[i] = sum_{t=0}^{KS-1} w[t] * x[i + t]
#   x shape: (N + KS - 1,)
#   y shape: (N,)
#   w shape: (KS,)

N  = getenv("N", 1 << 20)
KS = getenv("KS", 5)
BLOCK = getenv("BLOCK", 256)

def conv1d_uop_kernel(N, KS, BLOCK):
  assert N % BLOCK == 0, "for now, require N divisible by BLOCK"

  # global tensors
  y = UOp.placeholder((N,),          dtypes.float, slot=0)         # output
  x = UOp.placeholder((N+KS-1,),     dtypes.float, slot=1)         # input (padded)
  w = UOp.placeholder((KS,),         dtypes.float, slot=2)         # kernel

  # block/thread indices
  num_blocks = N // BLOCK
  blockIdx = UOp.special(num_blocks, "gidx0")
  tid      = UOp.special(BLOCK,      "lidx0")

  # logical output index i = blockIdx*BLOCK + tid
  i = blockIdx * BLOCK + tid

  # y tile view for (blockIdx, tid) → y[i]
  y_tiled = y.reshape(num_blocks, BLOCK)
  y_elem  = y_tiled[blockIdx, tid]

  # ---------------------------------
  # unrolled sum over kernel taps
  # y[i] = sum_{t=0}^{KS-1} w[t] * x[i + t]
  # ---------------------------------
  acc = 0.0
  for t in range(KS):
    acc = acc + x[i + t] * w[t]

  # store result
  sink = y_elem.store(acc)
  return sink.sink(arg=KernelInfo(name="conv1d_uop", opts_to_apply=())).simplify()

# ---------------------------
# simple test / benchmark
# ---------------------------

def test_conv1d_uop(N=1<<20, KS=5, BLOCK=256, runs=5):
  np.random.seed(0)

  # x padded: length N + KS - 1
  x_np = np.random.randn(N + KS - 1).astype(np.float32)
  w_np = np.random.randn(KS).astype(np.float32)

  # reference conv (valid)
  ref = np.empty(N, dtype=np.float32)
  for i in range(N):
    ref[i] = np.dot(x_np[i:i+KS], w_np)

  # ---------- UOp conv benchmark ----------
  print("=== UOp conv1d kernel ===")
  x = Tensor(x_np)
  w = Tensor(w_np)
  y = Tensor.empty(N, dtype=dtypes.float)

  # realize buffers so .uop.buffer exists
  Tensor.realize(x, w, y)

  sink = conv1d_uop_kernel(N, KS, BLOCK)
  bufs = [t.uop.buffer for t in [y, x, w]]
  ei = ExecItem(get_runner(Device.DEFAULT, sink), bufs)

  # warmup
  with Context(DEBUG=0):
    ei.run(wait=True)

  times = []
  with Context(DEBUG=0):
    for _ in range(runs):
      times.append(ei.run(wait=True))
  t_min = min(times)

  flops = 2.0 * N * KS
  gflops_uop = flops / t_min / 1e9
  print(f"UOp conv: time={t_min*1e3:.3f} ms, {gflops_uop:.2f} GFLOP/s")

  with Context(DEBUG=0):
    y_np = y.numpy()
  mse = np.mean((y_np - ref)**2)
  print("UOp conv MSE vs numpy:", mse)
  assert mse < 1e-6, "conv1d_uop result is wrong!"

  # ---------- nn.Conv1d benchmark ----------
  print("\n=== nn.Conv1d ===")

  # Input shape for Conv1d: (batch, in_channels, length)
  x_nn = Tensor(x_np.reshape(1, 1, -1))   # (1, 1, N+KS-1)

  # Create Conv1d layer: in_channels=1, out_channels=1, kernel_size=KS
  conv = nn.Conv1d(1, 1, KS, bias=False)

  # Try to set weights to match w_np for fairness (optional)
  try:
    w_tensor = Tensor(w_np.reshape(1, 1, KS))
    if hasattr(conv.weight, "assign"):
      conv.weight.assign(w_tensor)
    else:
      conv.weight = w_tensor
  except Exception as e:
    print("Warning: couldn't set conv.weight, using random weights instead:", e)

  # warmup
  GlobalCounters.reset()
  with Context(DEBUG=0):
    out = conv(x_nn)
    out.realize()

  # timing
  times_nn = []
  with Context(DEBUG=0):
    for _ in range(runs):
      t0 = time.perf_counter()
      out = conv(x_nn)
      out.realize()
      t1 = time.perf_counter()
      times_nn.append(t1 - t0)
  t_min_nn = min(times_nn)

  # Conv1d here: batch=1, out_ch=1, out_len=N, in_ch=1, kernel_size=KS
  # FLOPs = 2 * batch * out_ch * out_len * in_ch * KS
  flops_nn = 2.0 * 1 * 1 * N * 1 * KS
  gflops_nn = flops_nn / t_min_nn / 1e9
  print(f"nn.Conv1d: time={t_min_nn*1e3:.3f} ms, {gflops_nn:.2f} GFLOP/s")

  # optional correctness check if the output length matches
  with Context(DEBUG=0):
    out_np = out.numpy()    # expected shape (1,1,N)
  if out_np.shape == (1, 1, N):
    out_flat = out_np.reshape(N)
    mse_nn = np.mean((out_flat - ref)**2)
    print("nn.Conv1d MSE vs numpy:", mse_nn)
  else:
    print("nn.Conv1d output shape", out_np.shape, "!= (1,1,N), skipping MSE check.")

if __name__ == "__main__":
  test_conv1d_uop()
