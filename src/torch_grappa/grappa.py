from math import floor
from typing import Optional, Tuple

import torch
from jaxtyping import Float
from tqdm import tqdm

from .utils import grappa_index, matrix_batch_size, sympad, undo_sympad


def grappa(
    data: torch.Tensor,
    calib: torch.Tensor,
    R: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    lamda_tik: float = 0.0,
) -> torch.Tensor:
    """
    Simple 2D/3D GRAPPA algorithm for reconstructing undersampled k-space data.

    Requires:
    - 1 fully sampled dimension
    - Kernels with odd size in the fully sampled dimension(s), and even otherwise

    Parameters
    ----------
    data : Float[torch.Tensor, "... C *spatial"]
        Complex k-space data to be reconstructed.
        Should not include calibration region.
        The leading dimensions (if any) are treated as batch dimensions.
    calib : Float[torch.Tensor, "... C *calibspatial"]
        Calibration data used for GRAPPA reconstruction.
        Should be cropped to calibration region.
    R : Tuple[int, ...]
        Acceleration factor for each spatial dimension.
        At least dimension must be fully sampled (R=1) for this implementation.
    kernel_size : Tuple[int, ...]
        Size of the GRAPPA kernel for each spatial dimension.
        Must be odd in the fully sampled dimension and even in undersampled dimensions.
    lamda : float, optional
        Regularization (tikonov) parameter for GRAPPA, if desired. Default 0.
    batch_size : int, optional
        Batch size for processing large data. If None, processes all data at once.
        Useful for memory management with 3D / large kernels.

    Returns
    -------
    Float[torch.Tensor, "... C *spatial"]
        Reconstructed fully-sampled k-space data of shape (..., C, *spatial).
        Complex tensor.
    """

    ndim = len(kernel_size)
    assert (
        len(R) == ndim
    ), "R must be a tuple of integers matching the number of spatial dimensions."
    R = tuple([int(r) for r in R])

    fs_inds = torch.where(torch.tensor(R) == 1)[0]
    assert len(fs_inds) > 0, "At least one dimension must be fully sampled (R >= 1)."
    assert all(
        (kernel_size[i] % 2 == 1 if R[i] == 1 else kernel_size[i] % 2 == 0)
        for i in range(ndim)
    ), "Kernel size must be odd in fully sampled dimensions and even in undersampled dimensions."

    assert all(
        kernel_size[-i] < calib.shape[-i] for i in range(1, ndim + 1)
    ), "Calibration region must be larger than or equal to kernel size in all dimensions."

    # Prepare inputs
    batched = False
    if data.ndim > ndim + 1:
        batched = True
        n_batch_dims = data.ndim - (ndim + 1)
        batch_shape = data.shape[:n_batch_dims]
        assert (
            calib.shape[:n_batch_dims] == batch_shape
        ), "If batching input, calibration data must have the same leading dimensions."
        data = data.reshape(-1, *data.shape[n_batch_dims:])
        calib = calib.reshape(-1, *calib.shape[n_batch_dims:])
    else:
        data = data[None,]
        calib = calib[None,]

    input_fs_unordered = False
    if R[0] > 1:
        input_fs_unordered = True
        fs_dim = fs_inds[0].item()
        data = data.moveaxis(fs_dim + 2, 2)
        calib = calib.moveaxis(fs_dim + 2, 2)
        R = tuple([1] + list(R[:fs_dim]) + list(R[fs_dim + 1 :]))
        kernel_size = tuple(
            list(kernel_size[fs_dim : fs_dim + 1])
            + list(kernel_size[:fs_dim])
            + list(kernel_size[fs_dim + 1 :])
        )

    if ndim == 2:
        data = data[..., None]
        calib = calib[..., None]
        kernel_size = (kernel_size[0], kernel_size[1], 1)
        R = (R[0], R[1], 1)

    # Core call
    # TODO: properly integrate batching with batch size support.
    out = data.clone()
    for b in range(data.shape[0]):
        out[b] = _grappa(data[b], calib[b], kernel_size, R, lamda_tik)

    # Restore user dimensions
    if ndim == 2:
        out = out[..., 0]

    if input_fs_unordered:
        out = out.moveaxis(2, fs_dim + 2)

    if batched:
        out = out.reshape(*batch_shape, *out.shape[1:])
    else:
        out = out[0]

    return out


def _grappa(
    data: Float[torch.Tensor, "C Nx Ny Nz"],
    calib: Float[torch.Tensor, "C cx cy cz"],
    kernel_size: Tuple[int, int, int],
    R: Tuple[int, int, int],
    lamda_tik: float = 0.0,  # TODO: currently unused
) -> Float[torch.Tensor, "C Nx Ny Nz"]:
    """
    Core grappa algorithm with processed inputs.

    Parameters
    ----------
    data : Float[torch.Tensor, "C Nx Ny Nz"]
        Complex k-space data to be reconstructed.
        Should not include calibration region.
        Nx should be fully sampled.
    calib : Float[torch.Tensor, "C cx cy cz"]
        Calibration data used for GRAPPA reconstruction.
        Should be cropped to calibration region.

    Returns
    -------
    Float[torch.Tensor, "C Nx Ny Nz"]
        Reconstructed k-space data of shape (C, Nx, Ny, Nz).
        Complex tensor.
    """

    assert R[0] == 1, "The first dimension must be fully sampled (R[0] == 1)."

    pad = [int(floor(kernel_size[i] * R[i] / 2)) for i in range(3)]

    # Pad data
    data = sympad(data, pad)
    mask = data.abs() > 0

    # Flatten data for indexing
    in_shape = data.shape
    data = data.contiguous().view(-1)

    # Flatten calibration data
    calib_mask = torch.ones_like(calib, dtype=torch.bool)
    calib = calib.contiguous().view(-1)

    # Loop over kernels
    for kidx in range(R[1] * R[2] - 1):

        # Extract source and target indices from calibration mask
        src, tgt = grappa_index(kernel_size, calib_mask, pad, R, kidx)

        # Solve for GRAPPA weights via least squares
        weights = torch.linalg.lstsq(
            calib[src].T,
            calib[tgt].T,
        ).solution  # shape: (C * prod(kernel_size), C)

        # TODO: fix problem with inds taking too much memory (don't store for all coils?)
        # Extract source and target indices from undersampled data mask
        src, tgt = grappa_index(kernel_size, mask, pad, R, kidx)

        # Determine optimal batch size given remaining memory
        batch_size = matrix_batch_size(
            weights.shape[1],
            weights.shape[0],
            src.shape[1],
            dtype_bytes=data.element_size(),
            device=data.device,
        )

        # Apply GRAPPA weights to data
        if batch_size is not None and batch_size < src.shape[1]:
            Npoints = src.shape[1]
            for i in tqdm(
                range(0, Npoints, batch_size),
                desc=f"Processing kernel {kidx + 1}/{R[1] * R[2] - 1}",
                leave=False,
            ):
                batch_slc = slice(i, min(i + batch_size, Npoints))
                data[tgt[:, batch_slc]] = weights.T @ data[src[:, batch_slc]]
        else:
            data[tgt] = weights.T @ data[src]

        del src, tgt, weights
        torch.cuda.empty_cache()

    # reshape and unpad data
    data = data.view(*in_shape)
    data = undo_sympad(data, pad)

    return data
