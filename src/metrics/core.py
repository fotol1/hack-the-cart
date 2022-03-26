from typing import NamedTuple, Union
from abc import ABC, abstractmethod

import numpy as np


class PrepareTargetResult(NamedTuple):
    values: np.ndarray
    indices: np.ndarray


class Metric(ABC):
    """
    Abstract class with structure for all metrics in the project.
    """

    def __repr__(self) -> str:
        return "{:.4f}".format(self.get_metric_value())

    @abstractmethod
    def __call__(self, output: np.ndarray, target: np.ndarray) -> None:
        """
        Add new output and target to metric state.

        Parameters
        ----------
        output : `np.ndarray`, required
            Output from the model.
        target : `np.ndarray`, required
            True target.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_metric(self, reset: bool = False) -> Union[float, int]:
        """
        Get current metric value.

        Parameters
        ----------
        reset : `bool`, optional (default = `False`)
            Whether to reset metric state or not.
            If `True` then metric's sufficient statistics return to default state.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Reset metric's state."""
        raise NotImplementedError()


def validate_metric_inputs(output: np.ndarray, target: np.ndarray) -> None:
    if output.shape != target.shape:
        raise IndexError(
            "Unequal sizes for output and target: "
            f"output - {output.shape}, target - {target.shape}."
        )
    if not (np.equal(target, 0) | np.equal(target, 1)).all():
        raise ValueError("Target contains values outside of 0 and 1." f"\nTarget:\n{target}")


def prepare_target(
    output: np.ndarray, target: np.ndarray, return_indices: bool = False
) -> Union[np.ndarray, PrepareTargetResult]:
    """
    Sort target by output scores.

    Parameters
    ----------
    output : `np.ndarray`, required
        Model output scores for each item.
    target : `np.ndarray`, required
        True target for each item. It is a binary tensor.
    Returns
    -------
    `Union[np.ndarray, PrepareTargetResult]`
        Target sorted by output scores in descending order.
    """
    validate_metric_inputs(output, target)
    # Define order by sorted output scores.
    indices = np.argsort(-output, axis=-1)
    sorted_target = np.take_along_axis(target, indices=indices, axis=-1)
    return PrepareTargetResult(sorted_target, indices) if return_indices else sorted_target
