import gc

import torch


def solve_grappa_weights(
    src: torch.Tensor,
    tar: torch.Tensor,
    lamda_tik: float = 0.0,  # Tikhonov regularization parameter
) -> torch.Tensor:
    """
    Solve for GRAPPA weights using specified method.

    Parameters
    ----------
    src : torch.Tensor
        Source indices for GRAPPA kernel.
        Shape: (N_points, C*K)

    tar : torch.Tensor
        Target indices for GRAPPA kernel.
        Shape: (N_points, C)

    lamda_tik : float, optional
        Tikhonov regularization parameter. Default is 0.0 (no regularization).
        If greater than 0, will apply Tikhonov regularization and solve lstsq(AHA + lI, AHb).

    Returns
    -------
    torch.Tensor
        GRAPPA weights.
        Shape: (C*K, C)
    """

    if lamda_tik > 0:
        AHA = src.H @ src
        AHb = src.H @ tar

        # regularize
        if lamda_tik > 0:
            # normalize regularization scale by eigenvalues of A
            max_eig = torch.max(torch.linalg.eigvalsh(AHA)).item()
            AHA += torch.eye(src.shape[1], device=src.device) * lamda_tik * max_eig

        weights = torch.linalg.lstsq(AHA, AHb).solution

        del AHA, AHb
        torch.cuda.empty_cache()
        gc.collect()

    else:
        weights = torch.linalg.lstsq(src, tar).solution

    return weights
