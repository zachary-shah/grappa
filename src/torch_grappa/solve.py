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
            max_eig = power_method(AHA.abs(), tol=1e-6, max_iter=10)[0].item()

            AHA += torch.eye(src.shape[1], device=src.device) * lamda_tik * max_eig

        weights = torch.linalg.lstsq(AHA, AHb).solution

    else:
        weights = torch.linalg.lstsq(src, tar).solution

    return weights


def power_method(
    A: torch.Tensor, tol: float = 1e-6, max_iter: int = 10
) -> torch.Tensor:
    """
    Compute the largest eigenvalue and corresponding eigenvector of a matrix A
    using the power method.

    Parameters
    ----------
    A : torch.Tensor
        Input matrix of shape (N, N).
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 10.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the largest eigenvalue and the corresponding eigenvector.
        The eigenvalue is a scalar, and the eigenvector is a tensor of shape (N,).
    """

    evec = torch.randn(A.shape, device=A.device, dtype=A.dtype)
    eig_last = 0

    for i in range(max_iter):
        evec = A @ evec
        eig = evec.norm()
        evec /= eig
        if (eig - eig_last).abs() < tol * eig_last:
            break
        eig_last = eig

    return eig, evec
