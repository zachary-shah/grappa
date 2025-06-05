"""
Tools for sampling k-space in GRAPPA recon.
"""

from typing import Tuple

import torch


def segment_calibration(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, slice]:
    """
    Automatic segmentation of calibration data from k-space.

    Parameters
    ----------
    data : torch.Tensor
        K-space data with shape (C, *im_size), where N is the number of samples,
        C is the number of coils, and im_size is the size of the image in each dimension.
    return_slices : bool, optional
        If True, return the slices used for calibration, by default False.

    Returns
    -------
    calib : torch.Tensor
        Calibration data extracted from the k-space data.
    calib_slc : slice
        Slice object representing the calibration region in the k-space data.

    Use this function when your input still has the calibration region, such as:

    o o o o o o o o o o o o o
    . . . . . . . . . . . . .
    o o o o o o o o o o o o o
    . . . . . . . . . . . . .
    o o o o o o o o o o o o o
    . . . . o o o o . . . . .
    o o o o o o o o o o o o o
    . . . . o o o o . . . . .
    o o o o o o o o o o o o o
    . . . . . . . . . . . . .
    o o o o o o o o o o o o o
    . . . . . . . . . . . . .
    o o o o o o o o o o o o o
    """

    # Sampled points
    samp = (data.abs().norm(dim=0) > 0).to(torch.bool)

    im_size = samp.shape
    ndim = len(im_size)

    # Conver endpoints to matrix slices
    def get_slcs(css, ces, add_coil_dim=False):
        slc = tuple([slice(css[i], ces[i] + 1) for i in range(ndim)])
        if add_coil_dim:
            slc = (slice(None),) + slc
        return slc

    # Check that data is fully sampled in selected cal region
    def is_fs(css, ces):
        return samp[get_slcs(css, ces)].all()

    # intiial guess of calibration region as 3x3x3 center region
    css = [im_size[i] // 2 - 1 for i in range(ndim)]
    ces = [im_size[i] // 2 + 2 for i in range(ndim)]

    for d in range(ndim):
        # add left
        while (css[d] > 0) and is_fs(css, ces):
            css[d] -= 1
        if not is_fs(css, ces):
            css[d] += 1

        # add right
        while (ces[d] < im_size[d]) and is_fs(css, ces):
            ces[d] += 1
        if not is_fs(css, ces):
            ces[d] -= 1

    calib_slc = get_slcs(css, ces, add_coil_dim=True)

    calib = data[calib_slc].clone()

    return calib, calib_slc


def grappa_mask(im_size: Tuple[int, ...], R: Tuple[int, ...]) -> torch.Tensor:
    """
    Get a nd sampling mask for GRAPPA reconstruction.

    Parameters
    ----------
    im_size : Tuple[int, ...]
        Size of the image in each dimension.
    R : Tuple[int, ...]
        Acceleration factor in each dimension.
        R[i] > 1 means that the i-th dimension is under-sampled.

    Returns
    -------
    torch.Tensor
        A boolean mask of the same size as im_size, where True indicates sampled points.
        The mask is False in under-sampled dimensions according to R.
    """

    nd = len(im_size)
    assert len(R) == nd, "R must have the same number of dimensions as im_size"

    mask = torch.zeros(im_size, dtype=torch.long)
    for i in range(nd):
        slc = tuple(
            slice(None) if j != i else slice(0, im_size[i], R[i]) for j in range(nd)
        )
        mask[slc] += 1

    # Only keep intersection
    mask = mask == nd

    return mask
