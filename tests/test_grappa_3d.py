from time import perf_counter

import torch

# Optional install with `pip install pyeyes` for viz
from pyeyes import ComparativeViewer as cv

from torch_grappa.grappa import grappa
from torch_grappa.profiler import profile_decorator
from torch_grappa.sampling import grappa_mask
from torch_grappa.utils import fft, ifft

# Parameters
device = torch.device(5)  # GPU device to use
R = (1, 3, 2)
kernel_size = (5, 4, 2)
Ncal = (25, 25, 25)  # center region width for calib
noise_std = 1e-4
dataset = "head"  # head or brain
lamda_tik = 1e-6  # Tikhonov regularization parameter

if dataset == "head":
    data_path = "data/im_3d_head.pt"
elif dataset == "brain":
    data_path = "data/im_3d_brain.pt"
else:
    raise ValueError("Unknown dataset. Use 'head' or 'brain'.")

data = torch.load(
    data_path, weights_only=True, map_location=device if device.type != "cpu" else None
)
img = data["img"]
mps = data["mps"]  # Multi-coil sensitivity maps
# mask = data["mask"]  # brain mask

im_size = img.shape

# create sampling mask
samp_mask = grappa_mask(im_size, R).to(device)

"""
Simulate acquisition
"""
x = img[None,] * mps
y = fft(x, im_size)
y = y + torch.randn_like(y) * noise_std

calib_slc = [
    slice(im_size[i] // 2 - Ncal[i] // 2, im_size[i] // 2 + Ncal[i] // 2 + 1)
    for i in range(len(im_size))
]
calib_slc_full = tuple(
    [
        slice(
            None,
        )
    ]
    + calib_slc
)
calib_slc = tuple(calib_slc)
calib = y[calib_slc_full]

# Undersampling with calibration present
y_us = y * samp_mask[None,]
y_us[calib_slc_full] = calib


@profile_decorator(enable=True, verbose=True, save=True, save_path="./profs/3d_grappa")
def profile_loop(N=10):
    for _ in range(N):
        y_grappa = grappa(y_us, R, kernel_size, lamda_tik=lamda_tik)
    return y_grappa


N = 10
tstart = perf_counter()
print(f"Starting grappa with Rx={R[0]}, Ry={R[1]}, Rz={R[2]}.")
y_grappa = profile_loop(N)

print(f"Grappa took {(perf_counter() - tstart)/N:.2f} seconds")

x_us = ifft(y_us, im_size)
x_grappa = ifft(y_grappa, im_size)

nrmse = (x_grappa - x).abs().norm() / x.abs().norm()
print(f"NRMSE: {nrmse.item():.4f}")

img_rec = x.abs().norm(dim=0)
img_us = x_us.abs().norm(dim=0)  # * sqrt(prod(R))
img_grappa = x_grappa.abs().norm(dim=0)

# Visualization
cv(
    dict(
        gt=img_rec,
        us=img_us,
        grappa=img_grappa,
    ),
    list("xyz"),
    list("yz"),
).launch("Grappa Recons")
