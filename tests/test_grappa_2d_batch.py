from time import perf_counter

import torch

from torch_grappa.grappa import grappa
from torch_grappa.profiler import profile_decorator
from torch_grappa.sampling import segment_calibration
from torch_grappa.utils import ifft

# Parameters
Nx_calib = 31
kernel_size = (7, 4)
lamda_tik = 1e-6
device_idx = 1
device = torch.device(device_idx)

data = torch.load("data/epti_2d.pt", weights_only=False, map_location=device)
ksp_cart = data["ksp_cart"]  # Nt, Nz, Nc, X, Y
rec_gt = data["rec_gt"]  # Nt X Y
mps = data["mps"]  # Nc X Y
Ry = data["Ry"]
Nx = ksp_cart.shape[-2]
Ny = ksp_cart.shape[-1]

_, calib_slc = segment_calibration(ksp_cart[0, 0])
calib_slc_full = (
    slice(
        None,
    ),
    slice(
        None,
    ),
    *calib_slc,
)
ksp_calib_full = ksp_cart[calib_slc_full].clone()

# Reduced Nx region
Nx_slc = slice((Nx - Nx_calib) // 2, (Nx + Nx_calib) // 2 + 1)
ksp_calib = ksp_calib_full[:, :, :, Nx_slc, :].clone()

# Undersampled region
ksp_sampled = ksp_cart[..., ::Ry].clone()
ksp_cart = torch.zeros_like(ksp_cart)
ksp_cart[..., ::Ry] = ksp_sampled


# GRAPPA reconstructions
@profile_decorator(enable=True, verbose=True, save=True, save_path="./profs/grappa")
def profile_grappa(N=10):
    for _ in range(N):
        ksp_cart_out = grappa(
            ksp_cart,
            (1, Ry),
            kernel_size,
            calib=ksp_calib,
            lamda_tik=lamda_tik,
            batch_size=None,
        )
    return ksp_cart_out


tstart = perf_counter()
N = 10
ksp_cart = profile_grappa(N)
print(f"Avg time per call: {(perf_counter() - tstart)/N:.2f} s")
# Restore fully sampled region
ksp_cart[calib_slc_full] = ksp_calib_full

# Reconstruct time series of images
img_cart = ifft(ksp_cart, (Nx, Ny)).permute(0, 2, 3, 4, 1)[..., 0]  # Nt Nc Nx Ny
rec = torch.sum(img_cart * mps[None,].conj(), dim=1)

nrmse_val = (rec - rec_gt).norm() / rec_gt.norm()
print(f"NRMSE: {nrmse_val:.4f}")
