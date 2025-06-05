from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt
import torch

from torch_grappa.grappa import grappa
from torch_grappa.utils import fft, ifft

matplotlib.use("webagg")

# Parameters
R = (1, 3)
ncal = 24
kernel_size = (5, 2)
device = 3
N = 25  # repetitions for timing
lamda_tik = 1e-6  # Tikhonov regularization parameter
autocal = True  # Test automatic detection of calibration region from input data


def cc(x):
    return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=0))


# Test small 2D image with 8 coils
x = torch.load("data/im.pt", weights_only=True, map_location=torch.device(device))
im_size = x.shape[1:]
Nx, Ny = im_size

# Simulate
y = fft(x, im_size)
calib = y[:, (Nx - ncal) // 2 : (Nx + ncal) // 2, (Ny - ncal) // 2 : (Ny + ncal) // 2]
inp = y.clone() * 0
inp[:, :: R[0], :: R[1]] = y[:, :: R[0], :: R[1]]

if autocal:
    inp[:, (Nx - ncal) // 2 : (Nx + ncal) // 2, (Ny - ncal) // 2 : (Ny + ncal) // 2] = (
        calib
    )

ts = perf_counter()

for _ in range(N):
    if autocal:
        out = grappa(inp, R, kernel_size, lamda_tik=lamda_tik)
    else:
        out = grappa(inp, R, kernel_size, calib=calib, lamda_tik=lamda_tik)


te = perf_counter() - ts

print(f"GRAPPA took {te/N:.4f} seconds")
print(f"Total time for {N} runs: {te:.2f} seconds")

xhat = ifft(out, im_size)
xalias = ifft(inp, im_size)

nrmse = (xhat - x).norm() / x.norm()

print(f"NRMSE = {nrmse.item():.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(cc(x).cpu().numpy(), cmap="gray")
ax[1].imshow(cc(xalias).cpu().numpy(), cmap="gray")
ax[2].imshow(cc(xhat).cpu().numpy(), cmap="gray")
ax[0].set_title("Original")
ax[1].set_title("Aliased")
ax[2].set_title("Reconstructed")
plt.show()
