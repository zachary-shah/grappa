from typing import Tuple

import torch
from jaxtyping import Complex


def solve_grappa_weights(
    src: Complex[torch.Tensor, "B N_pt Nsrc"],
    tar: Complex[torch.Tensor, "B N_pt Ntar"],
    lamda_tik: float = 0.0,  # Tikhonov regularization parameter
) -> Complex[torch.Tensor, "B Nsrc Ntar"]:
    """
    Solve for GRAPPA weights on calibration data.

    Parameters
    ----------
    src : torch.Tensor
        Source data for GRAPPA kernel.
        Shape: (B, N_points, Nsrc)

    tar : torch.Tensor
        Target data for GRAPPA kernel.
        Shape: (B, N_points, Ntar)

    lamda_tik : float, optional
        Tikhonov regularization parameter. Default is 0.0 (no regularization).
        If > 0, will apply Tikhonov regularization and solve lstsq(AHA + lI, AHb).

    Returns
    -------
    torch.Tensor
        GRAPPA weights.
        Shape: (B, Nsrc, Ntar)
    """

    AHA = src.mH @ src
    AHb = src.mH @ tar

    if lamda_tik > 0:
        # normalize regularization scale by eigenvalues of A
        max_eig = power_method(AHA.abs(), tol=1e-6, max_iter=10)[0][:, None, None]
        AHA += torch.eye(src.shape[-1], device=src.device)[None,] * lamda_tik * max_eig

    weights = torch.linalg.lstsq(AHA, AHb).solution

    return weights


def power_method(
    A: Complex[torch.Tensor, "... N N"],
    tol: float = 1e-6,
    max_iter: int = 10,
) -> Tuple[Complex[torch.Tensor, "..."], Complex[torch.Tensor, "... N"]]:
    """
    Compute the largest eigenvalue and corresponding eigenvector of a matrix A
    using the power method.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (..., N, N).
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 10.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the largest eigenvalue and the corresponding eigenvector.
        The eigenvalue is shape (...), and the eigenvector is a tensor of shape (..., N,).
    """

    evec = torch.randn((*A.shape[:-1], 1), device=A.device, dtype=A.dtype)
    eig_last = torch.zeros(A.shape[:-2], device=A.device, dtype=A.dtype)

    for i in range(max_iter):
        evec = A @ evec
        eig = evec.norm(dim=-2, keepdim=True)
        evec /= eig
        if (eig - eig_last).abs().sum() < (tol * eig_last).abs().sum():
            break
        eig_last = eig

    eig = eig.squeeze((-1, -2))
    evec = evec.squeeze(-1)

    return eig, evec
