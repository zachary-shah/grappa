"""
Tools for sampling k-space in GRAPPA recon. 
"""
from typing import Tuple
import torch

# TODO: function to remove calibration region from data

def grappa_mask(
        im_size: Tuple[int, ...],
        R: Tuple[int, ...]) -> torch.Tensor:
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

    mask = torch.ones(im_size, dtype=torch.bool)
    for i in range(nd):
        if R[i] > 1:
            mask[tuple(slice(None) if j != i else slice(0, None, R[i]) for j in range(nd))] = False
    
    return mask
