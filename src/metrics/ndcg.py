from typing import List
import numpy as np
from src.metrics.core import Metric, prepare_target
from src.metrics.utils import exp_dcg, linear_dcg, non_empty_average


class NDCG(Metric):
    """
    Implementation of `Normalized Discounter Cumulative Gain` Metric.
    Graded relevance as a measure of  usefulness, or gain, from examining a set of items.
    Gain may be reduced at lower ranks.

    Parameters
    ----------
    topk : `int`, required
        Top-K elements to take into account.
    gain_function : `str`, optional (default = `"exp"`)
        Pick the gain function for the ground truth labels.
        Two options:
        - exp
        - linear
    """

    def __init__(self, topk: List[int], gain_function: str = "exp") -> None:
        super().__init__()
        assert all(x > 0 for x in topk), f"Invalid topk value: {topk}"
        assert gain_function in (
            "exp",
            "linear",
        ), f"Invalid gain_function value: {gain_function}"
        self._topk = topk
        self._total_ndcg = {f"ndcg@{k}": 0 for k in topk}
        self._total_count = 0
        self._dcg = exp_dcg if gain_function == "exp" else linear_dcg

    def _compute(self, output: np.ndarray, target: np.ndarray) -> float:
        topk = [min(target.shape[-1], x) - 1 for x in self._topk]
        # non_empty_sum ~ (batch)
        non_empty_sum = target.sum(axis=-1)
        # target_sorted_by_output ~ (users, items)
        target_sorted_by_output = prepare_target(output, target)
        ideal_target = prepare_target(target, target)
        # ideal_dcg ~ (users, items)
        ideal_dcg = self._dcg(ideal_target).cumsum(axis=-1)
        # prediction_dcg ~ (users, items)
        prediction_dcg = self._dcg(target_sorted_by_output).cumsum(axis=-1)
        # ndcg_score ~ (users, items)
        ndcg_score = prediction_dcg / ideal_dcg
        ndcg_score[ideal_dcg == 0] = 0.0
        # Compute an average excluding samples with all zeros
        # u - users, l - len(topk)
        mean_ndcg_score = non_empty_average(
            np.einsum("ul->lu", ndcg_score[:, topk]),
            batch_sum=non_empty_sum,
        )
        return mean_ndcg_score

    def __call__(self, output: np.ndarray, target: np.ndarray) -> None:
        self._total_count += 1
        values = self._compute(output, target)
        for topk, value in zip(self._topk, values):
            self._total_ndcg[f"ndcg@{topk}"] += value

    def get_metric(self, reset: bool = False) -> float:
        average_ndcg = {
            ndcg_k: value / self._total_count for ndcg_k, value in self._total_ndcg.items()
        }
        if reset:
            self.reset()
        return average_ndcg

    def reset(self) -> None:
        self._total_ndcg = {f"ndcg@{k}": 0 for k in self._topk}
        self._total_count = 0


if __name__ == "__main__":
    ndcg = NDCG(topk=[1, 3, 10, 100])
    inputs = {
        "output": np.array([[0.5, 0.4, 0.3, 0.2]]),
        "target": np.array([[1, 0, 1, 0]]),
    }
    answer = {
        "ndcg@1": 1.0,
        "ndcg@3": 0.91972,
        "ndcg@10": 0.91972,
        "ndcg@100": 0.91972,
    }
    ndcg(**inputs)
    print(ndcg.get_metric(reset=True))
    inputs_2 = {
        "output": np.array(
            [
                [9, 5, 3, 0, 7, 4, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 10],
                [0, 0, 1, 5, 9, 3, 0, 0, 0, 0, 0, 4, 0, 0, 10, 7, 0, 2, 8, 6],
                [0, 1, 4, 8, 6, 5, 3, 7, 10, 0, 9, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                [7, 8, 0, 0, 1, 0, 4, 0, 10, 0, 0, 6, 0, 0, 0, 9, 2, 3, 5, 0],
            ]
        ),
        "target": np.array(
            [
                [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            ]
        ),
    }
    answer_2 = {
        "ndcg@1": 0.25,
        "ndcg@3": 0.382680319,
        "ndcg@10": 0.418201509,
        "ndcg@100": 0.703113636,
    }
    ndcg(**inputs_2)
    print(ndcg.get_metric())
    for key, metric in ndcg.get_metric(reset=True).items():
        np.testing.assert_allclose(metric, answer_2[key])
