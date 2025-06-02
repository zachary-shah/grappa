import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from torch_grappa.utils import grappa_index

matplotlib.use("webagg")

# Test data from matlab implementation
pt = torch.load("data/index_test.pt", weights_only=False)

data = pt["data"]  # (8, 66, 72, 1)
mask = pt["mask"]  # (8, 66, 72, 1)

src_data = pt["src_data"]  # (8*prod(kernel_size), 2048)
trg_data = pt["trg_data"]  # (8, 2048)

src = pt["src"]  # (8*prod(kernel_size), 2048)
trg = pt["trg"]  # (8, 2048)

R = pt["R"]  # (1, 2, 2)
kernel_size = pt["kernel"]  # (3,)
pad = pt["pad"]  # (3,)
kernel_idx = pt["kernel_idx"]  # (1,)

# Test case - row-major
src_hat, trg_hat = grappa_index(kernel_size, mask, pad, R, kernel_idx)
data_flat = data.contiguous().view(-1)
src_data_hat = data_flat[src_hat]
tar_data_hat = data_flat[trg_hat]


# Need to sort, as ordering of points may be different
def eval(x, y):
    x_sorted = torch.sort(x.abs().flatten()).values
    y_sorted = torch.sort(y.abs().flatten()).values
    return x_sorted, y_sorted, (x_sorted - y_sorted).norm() / y_sorted.norm()


# Evaluate
src_data, src_data_hat, src_nrmse = eval(src_data, src_data_hat)
tar_data, tar_data_hat, tar_nrmse = eval(trg_data, tar_data_hat)

if src_nrmse > 1e-6:
    print(f"Test case failed: Src indexing NRMSE: {src_nrmse.item()}")
    p = src_data.shape[1] // 2
    plt.figure(figsize=(10, 5))
    plt.plot(np.abs(src_data.cpu().numpy()), label="matlab", color="red")
    plt.plot(np.abs(src_data_hat.cpu().numpy()), label="python", color="green")
    plt.legend()
    plt.title("Source Data Comparison")
    plt.show()
else:
    print("Test case passed: src!")

if tar_nrmse > 1e-6:
    print(f"Test case failed: Target indexing NRMSE: {tar_nrmse.item()}")
    p = trg_data.shape[0] // 2
    data_slc = slice(
        None,
    )
    plt.figure(figsize=(10, 5))
    plt.plot(trg_data.abs().cpu().numpy(), label="matlab", color="red")
    plt.plot(tar_data_hat.abs().cpu().numpy(), label="python", color="green")
    plt.legend()
    plt.title("Target Data Comparison")

else:
    print("Test case passed: tar!")

if (src_nrmse > 1e-6) or (tar_nrmse > 1e-6):
    plt.show()
else:
    print("All test cases passed!")
