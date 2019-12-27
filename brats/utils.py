import torch


def calculate_intersection(x1: torch.Tensor, x2: torch.Tensor,
                           dim=None) -> torch.Tensor:
    """
    Calculates intersection of two tensors.
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimensions along which the intersection will be summed
    Returns:
        torch.Tensor: Tensor containing the intersection. Dimensionality
            may differ, depending on the dim argument
    """
    return (x1 * x2).sum(dim=dim) if dim is not None else (x1 * x2).sum()


def calculate_union(x1: torch.Tensor, x2: torch.Tensor,
                    dim=None) -> torch.Tensor:
    """
    Calculates the union of two tensors
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimensions along which the intersection will be summed
    Returns:
        torch.Tensor: Tensor containing the union. Dimensionality may
            differ, depending on the dim argument
    """
    return x1.sum(dim=dim) + x2.sum(dim=dim) if dim is not None \
        else x1.sum() + x2.sum()
