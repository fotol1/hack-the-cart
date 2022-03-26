from typing import List
import numpy as np
from src.metrics.core import Metric, prepare_target
from src.metrics.utils import non_empty_average, nan_to_num


class Recall(Metric):
    def __init__(self, topk: List[int]) -> None:
        assert all(x > 0 for x in topk), f"Invalid topk value: {topk}"
        self._total_recall = {f"recall@{k}": 0 for k in topk}
        self._total_count = 0
        self._topk = topk

    def _compute(self, output: np.ndarray, target: np.ndarray) -> float:
        topk = [min(target.shape[-1], x) - 1 for x in self._topk]
        # non_empty_sum ~ (batch)
        non_empty_sum = target.sum(axis=-1)
        # target_sorted_by_output ~ (users, topk)
        target_sorted_by_output = prepare_target(output, target)
        recall_score = np.einsum(
            "ul->lu", target_sorted_by_output.cumsum(axis=-1)[:, topk]
        ) / target.sum(axis=-1)
        # Compute an average excluding samples with all zeros
        # recall_score ~ (topk)
        recall_score = non_empty_average(nan_to_num(recall_score), batch_sum=non_empty_sum)
        return recall_score

    def __call__(self, output: np.ndarray, target: np.ndarray) -> None:
        self._total_count += 1
        values = self._compute(output, target)
        for topk, recall in zip(self._topk, values):
            self._total_recall[f"recall@{topk}"] += recall

    def get_metric(self, reset: bool = False) -> float:
        average_recall = {
            recall_k: value / self._total_count for recall_k, value in self._total_recall.items()
        }
        if reset:
            self.reset()
        return average_recall

    def reset(self) -> None:
        self._total_recall = {f"recall@{k}": 0 for k in self._topk}
        self._total_count = 0


if __name__ == "__main__":
    recall = Recall(topk=[1, 3, 10, 100])
    inputs = {
        "output": np.array([[0.5, 0.4, 0.3, 0.2]]),
        "target": np.array([[1, 0, 1, 0]]),
    }
    recall(**inputs)
    print(recall.get_metric(reset=True))
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
        "recall@1": (1 / 8) / 4,
        "recall@3": (2 / 8 + 1 / 8 + 1 / 10 + 1 / 9) / 4,
        "recall@10": (5 / 8 + 3 / 8 + 5 / 10 + 3 / 9) / 4,
        "recall@100": 1.0,
    }
    recall(**inputs_2)
    print(recall.get_metric())
    assert answer_2 == recall.get_metric(reset=True)
