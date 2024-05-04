from typing import List

import torch
import torch.nn.functional as F

"""

An implementation of the parallel scan operation in PyTorch (Blelloch version).
Please see docs/pscan.ipynb for a detailed explanation of what happens here.

"""

def log2(x: float) -> float:
    return torch.log(x) / torch.log(2.0)

def npo2(len:int):
    """
    Returns the next power of 2 above len
    """

    return 2 ** torch.ceil(log2(float(len)))


def pad_npo2(X):
    """
    Pads input length dim to the next power of 2

    Args:
        X : (B, L, D, N)

    Returns:
        Y : (B, npo2(L), D, N)
    """

    len_npo2 = npo2(X.size(1))
    pad_tuple: List[int] = [0, 0, 0, 0, 0, int(len_npo2 - X.size(1))]
    return F.pad(X, pad_tuple, "constant", 0.0)

@torch.jit.script
def pscan(A: torch.Tensor, X: torch.Tensor):
    # A : (B, D, L, N)
    # X : (B, D, L, N)

    # modifies X in place by doing a parallel scan.
    # more formally, X will be populated by these values :
    # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

    # only supports L that is a power of two (mainly for a clearer code)

    B, D, L, _ = A.size()
    num_steps = int(log2(float(L)))

    # up sweep (last 2 steps unfolded)
    Aa = A
    Xa = X
    for _ in range(num_steps - 2):
        T = Xa.size(2)
        Aa = Aa.view(B, D, T // 2, 2, -1)
        Xa = Xa.view(B, D, T // 2, 2, -1)

        Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
        Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

        Aa = Aa[:, :, :, 1]
        Xa = Xa[:, :, :, 1]

    # we have only 4, 2 or 1 nodes left
    if Xa.size(2) == 4:
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])

        Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
    elif Xa.size(2) == 2:
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        return
    else:
        return

    # down sweep (first 2 steps unfolded)
    Aa = A[:, :, int(2 ** (num_steps - 2) - 1):L:int(2 ** (num_steps - 2))]
    Xa = X[:, :, int(2 ** (num_steps - 2) - 1):L:int(2 ** (num_steps - 2))]
    Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
    Aa[:, :, 2].mul_(Aa[:, :, 1])

    for k in range(num_steps - 3, -1, -1):
        Aa = A[:, :, int(2 ** k - 1):L:int(2 ** k)]
        Xa = X[:, :, int(2 ** k - 1):L:int(2 ** k)]

        T = Xa.size(2)
        Aa = Aa.view(B, D, T // 2, 2, -1)
        Xa = Xa.view(B, D, T // 2, 2, -1)

        Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
        Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

@torch.jit.script
def pscan_forward(A_in: torch.Tensor, X_in: torch.Tensor) -> torch.Tensor:
    """
    Applies the parallel scan operation, as defined above. Returns a new tensor.
    If you can, privilege sequence lengths that are powers of two.

    Args:
        A_in : (B, L, D, N)
        X_in : (B, L, D, N)

    Returns:
        H : (B, L, D, N)
    """

    L = X_in.size(1)

    # cloning is requiered because of the in-place ops
    if L == npo2(L):
        A = A_in.clone()
        X = X_in.clone()
    else:
        # pad tensors (and clone btw)
        A = pad_npo2(A_in)  # (B, npo2(L), D, N)
        X = pad_npo2(X_in)  # (B, npo2(L), D, N)

    # prepare tensors
    A = A.transpose(2, 1)  # (B, D, npo2(L), N)
    X = X.transpose(2, 1)  # (B, D, npo2(L), N)

    # parallel scan (modifies X in-place)
    pscan(A, X)

    # slice [:, :L] (cut if there was padding)
    return X.transpose(2, 1)[:, :L]
