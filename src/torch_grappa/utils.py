from math import ceil
from typing import Tuple

import torch
from jaxtyping import Float


def fft_resize(input: torch.tensor, oshape: tuple):
    """Resize with zero-padding or cropping.

    Args:
        input (torch.tensor): Input array.
        oshape (tuple of ints): Output shape.

    Returns:
        torch.tensor: Zero-padded or cropped result.
    """

    assert len(input.shape) == len(
        oshape
    ), "Input and output must have same number of dimensions."

    ishape = input.shape

    if ishape == oshape:
        return input

    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape, oshape)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape, ishift, oshape, oshift)
    ]

    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape, dtype=input.dtype)
    output[oslice] = input[islice]

    return output


def fft(x: torch.Tensor, o_im_shape: tuple, center: bool = True) -> torch.Tensor:
    """
    Compute the cartesian iFFT of image data with shape (..., im_shape)

    Parameters
    ----------
    x : torch.Tensor
        Input data with shape (..., in_img_shape)
    o_im_shape : tuple
        Desired output image shape
    center : bool, optional
        Whether to center the transform, by default True

    Returns
    -------
    torch.Tensor
        Output data with shape (..., o_im_shape)
    """

    fftdims = tuple(range(-len(o_im_shape), 0))
    newshape = (*x.shape[: -len(o_im_shape)], *o_im_shape)

    if center:
        x = fft_resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm="ortho")
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm="ortho")

    return x


def ifft(x: torch.Tensor, o_im_shape: tuple, center: bool = True) -> torch.Tensor:
    """
    Compute the cartesian iFFT of image data with shape (..., im_shape)

    Parameters
    ----------
    x : torch.tensor
        Input data with shape (..., in_shape)
    o_im_shape : tuple
        Desired output image shape
    center : bool, optional
        Whether to center the transform, by default True

    Returns
    -------
    torch.Tensor
        Output data with shape (..., o_im_shape)
    """

    fftdims = tuple(range(-len(o_im_shape), 0))
    newshape = (*x.shape[: -len(o_im_shape)], *o_im_shape)

    if center:
        x = fft_resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm="ortho")
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm="ortho")

    return x


def sympad(
    data: torch.Tensor,
    pad: Tuple[int, ...],
) -> torch.Tensor:
    """
    Zero pad input in kspace directions.

    Parameters
    ----------
    data : Float[torch.Tensor, "... *spatial"]
        complex kspace
    pad : Tuple[int, ...]
        Padding per spatial dimension. Applied symmetrically to each side of tensor.

    Returns
    -------
    Float[torch.Tensor, "... *spatial"]
        Tensor which is zero padded in spatial dimensions.
    """

    pad_tuple = tuple(pad)
    pad_args = []
    for p in reversed(pad_tuple):
        pad_args.extend([p, p])

    return torch.nn.functional.pad(data, pad_args, mode="constant", value=0.0).to(
        data.dtype
    )


def undo_sympad(
    data: torch.Tensor,
    pad: Tuple[int, ...],
) -> torch.Tensor:
    """
    Remove zero padding from input in kspace directions.

    Parameters
    ----------
    data : Float[torch.Tensor, "... *spatial"]
        complex kspace
    pad : Tuple[int, ...]
        Padding per spatial dimension. Applied symmetrically to each side of tensor.

    Returns
    -------
    Float[torch.Tensor, "... *spatial"]
        Tensor with zero padding removed in spatial dimensions.
    """

    pad_tuple = tuple(pad)

    slc = []
    for i in range(data.ndim - len(pad_tuple)):
        slc.append(slice(None))
    for p in pad_tuple:
        slc.append(slice(p, -p) if p > 0 else slice(None))
    slc = tuple(slc)
    return data[slc]


def grappa_index(
    kernel_size: Tuple[int, int, int],
    samp: Float[torch.Tensor, "C Nx Ny Nz"],
    pad: Tuple[int, int, int],
    R: Tuple[int, int, int],
    kernel_idx: int,
    return_single_coil: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get linearized indices for source and target data.

    Parameters
    ----------
    kernel_size : Tuple[int, int, int]
        Size of the GRAPPA kernel for each spatial dimension.
    samp : Float[torch.Tensor, "C Nx Ny Nz"]
        Sampling mask, should be torch float tensor
    pad : Tuple[int, int, int]
        Padding applied to the data. Same size as spatial dimensions (symmetric padding on each side per axis).
    R : Tuple[int, int, int]
        Acceleration factors for each spatial dimension.
        Will be 1 for first dimension (fully sampled dim).
    kernel_idx : int
        Indicates which of the prod(R)-1 kernels to use.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor], where:
    - First tensor contains source indices for Grappa kernel, such that indexing
      into (C, Nx, Ny, Nz) data will give (C, inds) of the source data.
    - Second tensor contains target indices for Grappa kernel, such that indexing
      into (C, Nx, Ny, Nz) data witll give (C*K, inds) of the target data, where K is prod(kernel_size)

    Usage:
    ```
    data = torch.randn(Nc, Nx, Ny, Nz) # data array
    data_flat = data.contiguous().view(-1) # flattened row-major array
    src, tar = grappa_index(kernel_size, samp, pad, R, kernel_idx)
    src_data = data_flat[src] # (Nc*K, N_points)
    tar_data = data_flat[tar] # (Nc, N_points)
    ```

    """

    Nc, Nx, Ny, Nz = samp.shape
    device = samp.device

    # Ensure only y and z are undersampled
    assert (
        R[0] == 1
    ), "First spatial dimension must be fully sampled in `grappa_index` call."
    assert kernel_idx < (
        R[1] * R[2] - 1
    ), "kernel_idx must be less than R[1] * R[2] - 1."

    # 1) Compute all single-coil source inds in kernel
    src_lins = [
        torch.arange(0, R[i] * kernel_size[i], R[i], device=device, dtype=torch.long)
        for i in range(3)
    ]
    XX, YY, ZZ = torch.meshgrid(*src_lins, indexing="ij")
    k_src = (ZZ + YY * Nz + XX * Nz * Ny).flatten()

    # 2) Get the target index for this kernel
    ytyp = (kernel_idx + 1) % R[1]
    ztyp = (kernel_idx + 1) // R[1]
    kernel_ofs = [0, ytyp, ztyp]
    k_tar = [R[i] * (ceil(kernel_size[i] / 2) - 1) + kernel_ofs[i] for i in range(3)]
    k_tar = k_tar[2] + k_tar[1] * Nz + k_tar[0] * Nz * Ny

    # 3) Subtract target from source to get relative to targets
    k_src -= k_tar

    # 4) Find all possible target indices from first coil mask
    sub = undo_sympad(samp[0], pad)
    sub_shifted = torch.roll(sub, shifts=(0, ytyp, ztyp), dims=(0, 1, 2))
    sub_padded = sympad(sub_shifted, pad)
    tar_crd = torch.nonzero(sub_padded, as_tuple=False).T
    tar_single_coil = tar_crd[2] + tar_crd[1] * Nz + tar_crd[0] * Nz * Ny

    # 5) Create source indices relative to target indices
    src_single_coil = tar_single_coil[None,] + k_src[:, None]

    # 6) Replictate for all coils
    coil_ofs = torch.arange(Nc, device=device, dtype=torch.long) * Nx * Ny * Nz

    if return_single_coil:
        # For memory efficiency
        return src_single_coil, tar_single_coil, coil_ofs
    else:
        return index_expand_coils(src_single_coil, tar_single_coil, coil_ofs)


def index_expand_coils(
    src_single: torch.LongTensor,
    tar_single: torch.LongTensor,
    coil_ofs: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Epxand source and target indices to include coil offsets.
    """

    tar = coil_ofs[:, None] + tar_single[None, :]

    # Flatten coil / kernel dimension together
    src = coil_ofs[:, None, None] + src_single[None, :, :]
    src = src.view(-1, src.shape[-1])

    return src, tar


def index_batch_size(
    src_single: torch.LongTensor,
    coil_ofs: torch.LongTensor,
    data: torch.Tensor,
) -> int:
    """
    Determine batch size for applying GRAPPA given remaining memory.

    Required memory:

    Source inds: Nsrc * P * idx_bytes
    Target inds: Ntar * P * idx_bytes

    Src data: Nsrc * P * data_bytes
    Target data: Ntar * P * data_bytes

    Returns int batch size, or None if no batching is possible.
    """

    device = data.device
    if not torch.cuda.is_available() or device == "cpu":
        return None

    idx_bytes = src_single.element_size()
    data_bytes = data.element_size()
    total_bytes = idx_bytes + data_bytes

    Ntar = coil_ofs.shape[0]
    Nsrc = src_single.shape[0] * Ntar

    size_avail = torch.cuda.mem_get_info(device)[0]
    size_single = (Nsrc + Ntar * 2) * total_bytes
    if size_avail <= 0 or size_single <= 0:
        print("No memory available for batching.")
        return None

    batch_size = size_avail // size_single

    return batch_size
