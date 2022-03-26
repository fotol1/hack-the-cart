from typing import Union, List
import numpy as np


def nan_to_num(tensor: np.ndarray, nan: float = 0.0) -> np.ndarray:
    return np.where(
        np.logical_or(np.isnan(tensor), np.isinf(tensor)),
        np.full_like(tensor, fill_value=nan),
        tensor,
    )


def exp_dcg(tensor: np.ndarray) -> np.ndarray:
    """Calculate `Exponential` gain function for `NDCG` Metric."""
    gains = (2**tensor) - 1
    discounts = 1 / np.log2(np.arange(0, tensor.shape[-1], dtype=np.float32) + 2.0)
    return gains * discounts


def linear_dcg(tensor: np.ndarray) -> np.ndarray:
    """Calculate `Linear` gain function for `NDCG` Metric."""
    discounts = 1 / np.log2(np.arange(0, tensor.shape[-1], dtype=np.float32) + 1.0)
    discounts[0] = 1
    return tensor * discounts


def non_empty_average(
    x: np.ndarray, batch_sum: np.ndarray, axis: Union[int, List[int]] = -1
) -> np.ndarray:
    num_non_empty_seq = (batch_sum > 0).astype(np.float32).sum().clip(min=1e-13)
    return nan_to_num((x / num_non_empty_seq)).sum(axis=axis)
