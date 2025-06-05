from math import floor, prod
from typing import Optional, Tuple

import torch
from jaxtyping import Complex
from tqdm import tqdm

from .sampling import segment_calibration
from .solve import solve_grappa_weights
from .utils import (
    batch_iterator,
    grappa_index,
    index_batch_size,
    index_expand_coils,
    sympad,
    undo_sympad,
)


def grappa(
    data: torch.Tensor,
    R: Tuple[int, ...],
    kernel_size: Tuple[int, ...],
    calib: Optional[torch.Tensor] = None,
    lamda_tik: float = 0.0,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Simple 2D/3D GRAPPA algorithm for reconstructing undersampled k-space data.

    Requirements:
    - 1 fully sampled dimension
    - Kernels with odd size in the fully sampled dimension(s), and even otherwise
    - If batching, the sampling pattern should be the same across batches.
    - TODO: shared calibration option, where all items in batch use shared weights.

    Parameters
    ----------
    data : Float[torch.Tensor, "... C *spatial"]
        Complex k-space data to be reconstructed.
        Should not include calibration region if calib is provided.
        The leading dimensions (if any) are treated as batch dimensions.
    R : Tuple[int, ...]
        Acceleration factor for each spatial dimension.
        At least dimension must be fully sampled (R=1) for this implementation.
    kernel_size : Tuple[int, ...]
        Size of the GRAPPA kernel for each spatial dimension.
        Must be odd in the fully sampled dimension and even in undersampled dimensions.
    calib : Float[torch.Tensor, "... C *calibspatial"]
        Calibration data used for GRAPPA reconstruction.
        Should be cropped to calibration region.
        If not provided, will try to auto-segment the calibration region from the data.
    lamda_tik : float, optional
        Regularization (tikonov) parameter for GRAPPA, if desired. Default 0.
    batch_size : Optional[int], optional
        Batch size across leading dimensions of data.
        For parallelizing across many small GRAPPA problems, can result in 2x speedup.
        If None, will parallelize over all batches at once.

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

    input_calib = False
    if calib is not None:
        input_calib = True
        assert all(
            kernel_size[-i] < calib.shape[-i] for i in range(1, ndim + 1)
        ), "Calibration region must be larger than or equal to kernel size in all dimensions."

    # Prepare inputs
    batched = False
    if data.ndim > ndim + 1:
        batched = True
        n_batch_dims = data.ndim - (ndim + 1)
        batch_shape = data.shape[:n_batch_dims]
        data = data.reshape(-1, *data.shape[n_batch_dims:])
        if input_calib:
            assert (
                calib.shape[:n_batch_dims] == batch_shape
            ), "If batching input and providing calibration, calibration data must have the same leading dimensions."
            calib = calib.reshape(-1, *calib.shape[n_batch_dims:])
    else:
        data = data[None,]
        if input_calib:
            calib = calib[None,]

    input_fs_unordered = False
    if R[0] > 1:
        input_fs_unordered = True
        fs_dim = fs_inds[0].item()
        data = data.moveaxis(fs_dim + 2, 2)
        if input_calib:
            calib = calib.moveaxis(fs_dim + 2, 2)
        R = tuple([1] + list(R[:fs_dim]) + list(R[fs_dim + 1 :]))
        kernel_size = tuple(
            list(kernel_size[fs_dim : fs_dim + 1])
            + list(kernel_size[:fs_dim])
            + list(kernel_size[fs_dim + 1 :])
        )

    if ndim == 2:
        data = data[..., None]
        if input_calib:
            calib = calib[..., None]
        kernel_size = (kernel_size[0], kernel_size[1], 1)
        R = (R[0], R[1], 1)

    # get calib region if needed
    if not input_calib:
        calib_slc = segment_calibration(data.abs().sum(dim=0))[1]

    B = data.shape[0]
    if batch_size is not None:
        batch_size = max(min(batch_size, B), 1)
    else:
        batch_size = B

    # Core call
    out = data.clone()
    for bslc in tqdm(
        batch_iterator(B, batch_size),
        desc="GRAPPA batches",
        leave=False,
        disable=not batched,
    ):

        if not input_calib:
            calib_batch = data[(bslc,) + calib_slc].clone()
        else:
            calib_batch = calib[bslc]

        out[bslc] = _grappa(
            data[bslc], calib_batch, kernel_size, R, lamda_tik, verbose=not batched
        )

        # Ensure consistenty with calib
        if not input_calib:
            out[(bslc,) + calib_slc] = calib_batch

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
    data: Complex[torch.Tensor, "B C Nx Ny Nz"],
    calib: Complex[torch.Tensor, "B C cx cy cz"],
    kernel_size: Tuple[int, int, int],
    R: Tuple[int, int, int],
    lamda_tik: float = 0.0,
    verbose: bool = True,
) -> Complex[torch.Tensor, "B C Nx Ny Nz"]:
    """
    Core grappa algorithm with processed inputs.

    Parameters
    ----------
    data : Float[torch.Tensor, "B C Nx Ny Nz"]
        Complex k-space data to be reconstructed.
        Nx should be fully sampled.
    calib : Float[torch.Tensor, "B C cx cy cz"]
        Calibration data used for GRAPPA reconstruction.
        Should be cropped to calibration region.

    Returns
    -------
    Float[torch.Tensor, "B C Nx Ny Nz"]
        Reconstructed k-space data of shape (B, C, Nx, Ny, Nz).
        Complex tensor.
    """

    assert R[0] == 1, "The first dimension must be fully sampled (R[0] == 1)."

    pad = [int(floor(kernel_size[i] * R[i] / 2)) for i in range(3)]

    # Pad data
    data = sympad(data, pad)
    mask = data.abs().sum(dim=0) > 0

    # Flatten data for indexing
    B, in_shape = data.shape[0], data.shape[1:]
    data = data.contiguous().view(B, -1)

    # Flatten calibration data
    calib_mask = torch.ones_like(calib[0], dtype=torch.bool)
    calib = calib.contiguous().view(B, -1)

    # Loop over kernels
    Nk = prod(R) - 1
    for kidx in tqdm(range(Nk), desc="GRAPPA kernel", leave=False, disable=not verbose):

        # Extract source and target indices from calibration mask
        src, tar = grappa_index(kernel_size, calib_mask, pad, R, kidx)

        # Solve for GRAPPA weights with regularized least squares: (B, C*K, C)
        weights = solve_grappa_weights(
            calib[:, src].mT, calib[:, tar].mT, lamda_tik=lamda_tik
        )

        # Extract source and target indices from undersampled data mask
        src_sc, tar_sc, coil_ofs = grappa_index(
            kernel_size, mask, pad, R, kidx, return_single_coil=True
        )

        # Determine optimal batch size given remaining memory
        batch_size = index_batch_size(src_sc, coil_ofs, data)

        # Apply GRAPPA weights to data
        if batch_size and (0 < batch_size < src_sc.shape[1]):
            Npoints = src_sc.shape[1]
            for i in range(0, Npoints, batch_size):
                batch_slc = slice(i, min(i + batch_size, Npoints))
                src_batch, tar_batch = index_expand_coils(
                    src_sc[:, batch_slc], tar_sc[batch_slc], coil_ofs
                )
                data[:, tar_batch] = weights.mT @ data[:, src_batch]
        else:
            src, tar = index_expand_coils(src_sc, tar_sc, coil_ofs)
            data[:, tar] = weights.mT @ data[:, src]

        del src, tar, weights
        torch.cuda.empty_cache()

    # reshape and unpad data
    data = data.view(B, *in_shape)
    data = undo_sympad(data, pad)

    return data
