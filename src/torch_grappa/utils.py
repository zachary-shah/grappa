from typing import Tuple
from jaxtyping import Float
from math import ceil

import torch

def matrix_batch_size(
        M: int,
        N: int,
        P: int,
        dtype_bytes: int = 8, 
        device: torch.device = "cpu") -> int:
    """
    Given a problem of solving Y = A @ B, where A is MxN, B is NxP, and Y is MxP,
    compute some batch_size Pb <= P such that Y[:, :Pb = A @ B[:, :Pb] can be solved
    with remaining memory.
    """

    if device == "cpu" or (not torch.cuda.is_available()):
        return P # no batching on CPU

    size_avail = torch.cuda.mem_get_info(device)[0] // dtype_bytes - (M * N)
    size_per_batch = M * 2 + N
    if size_avail <= 0 or size_per_batch <= 0:
        return 0
    
    return size_avail // size_per_batch


def fft_resize(input: torch.tensor, 
                oshape: tuple):
    """Resize with zero-padding or cropping.

    Args:
        input (torch.tensor): Input array.
        oshape (tuple of ints): Output shape.

    Returns:
        torch.tensor: Zero-padded or cropped result.
    """

    assert len(input.shape) == len(oshape), \
        "Input and output must have same number of dimensions."
    
    ishape = input.shape

    if ishape == oshape: return input

    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape, oshape)]
    
    copy_shape = [min(i - si, o - so)
                  for i, si, o, so in zip(ishape, ishift, oshape, oshift)]
        
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape, dtype=input.dtype)
    output[oslice] = input[islice]

    return output


def fft(x: torch.Tensor, 
         o_im_shape: tuple, 
         center: bool = True) -> torch.Tensor:
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

    fftdims = tuple(range(-len(o_im_shape),0))
    newshape = (*x.shape[:-len(o_im_shape)], *o_im_shape)
    
    if center:
        x = fft_resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm='ortho')
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm='ortho')

    return x


def ifft(x: torch.Tensor, 
         o_im_shape: tuple, 
         center: bool = True) -> torch.Tensor:
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

    fftdims = tuple(range(-len(o_im_shape),0))
    newshape = (*x.shape[:-len(o_im_shape)], *o_im_shape)
    
    if center:
        x = fft_resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm='ortho')
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm='ortho')

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
    
    return torch.nn.functional.pad(data, pad_args, mode="constant", value=0.0).to(data.dtype)


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
    src, trg = grappa_index(kernel_size, samp, pad, R, kernel_idx)
    src_data = data_flat[src] # (Nc*K, N_points)
    trg_data = data_flat[trg] # (Nc, N_points)
    ```

    """

    Nc, Nx, Ny, Nz = samp.shape
    device = samp.device
    
    # Ensure only y and z are undersampled
    assert R[0] == 1, "First spatial dimension must be fully sampled in `grappa_index` call."
    assert kernel_idx < (R[1] * R[2] - 1), "kernel_idx must be less than R[1] * R[2] - 1."

    # 1) Compute all single-coil source inds in kernel
    src_lins = [torch.arange(0, R[i] * kernel_size[i], R[i], device=device, dtype=torch.long) for i in range(3)]
    XX, YY, ZZ = torch.meshgrid(*src_lins, indexing='ij')
    k_src = (ZZ + YY * Nz + XX * Nz * Ny).flatten()

    # 2) Get the target index for this kernel
    ytyp = (kernel_idx + 1) %  R[1]
    ztyp = (kernel_idx + 1) // R[1]
    kernel_ofs = [0, ytyp, ztyp]
    k_tar = [R[i] * (ceil(kernel_size[i] / 2) - 1) + kernel_ofs[i] for i in range(3)]
    k_trg = k_tar[2] + k_tar[1] * Nz + k_tar[0] * Nz * Ny

    # 3) Subtract target from source to get relative to targets
    k_src -= k_trg

    # 4) Find all possible target indices from first coil mask
    sub = undo_sympad(samp[0], pad)
    sub_shifted = torch.roll(sub, shifts=(0, ytyp, ztyp), dims=(0, 1, 2))
    sub_padded = sympad(sub_shifted, pad)
    trg_crd = torch.nonzero(sub_padded, as_tuple=False).T
    trg_single_coil = trg_crd[2] + trg_crd[1] * Nz + trg_crd[0] * Nz * Ny

    # 5) Create source indices relative to target indices
    src_single_coil = trg_single_coil[None,] + k_src[:, None]

    # 6) Replictate for all coils
    coil_ofs = torch.arange(Nc, device=device, dtype=torch.long) * Nx * Ny * Nz
    trg = coil_ofs[:, None] + trg_single_coil[None, :] # (Nc, Npix)
    src = coil_ofs[:, None, None] + src_single_coil[None, :, :] # (Nc, K, Npix)

    # Flatten to (Nc * prod(kernel_size), Npix)
    src = src.view(-1, src.shape[-1])  

    return src.long(), trg.long()
