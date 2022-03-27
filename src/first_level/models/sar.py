from typing import List, Dict, Any
import numpy as np
from scipy import sparse
from copy import deepcopy
from src.metrics import NDCG, Recall
from src.first_level.models.core import Model
from src.first_level.models.utils import jaccard, lift


SIMILARITY_FUNC = {
    "jaccard": jaccard,
    "lift": lift,
    "cooccurrence": lambda x: np.array(x),
}


class SAR(Model):
    def __init__(
        self,
        similarity: str = "jaccard",
        threshold: float = 1.0,
        metrics_topk: List[int] = [20, 50, 100],
    ) -> None:
        assert similarity in ("jaccard", "lift", "cooccurrence")
        self._threshold = threshold
        self._train_matrix = None
        self._item_similarity = None
        self._item_frequencies = None
        self._similarity_func = SIMILARITY_FUNC[similarity]
        self._metrics = [
            NDCG(metrics_topk),
            Recall(metrics_topk),
        ]

    def _compute_cooccurrence_matrix(self, matrix: sparse.csr_matrix) -> None:
        matrix = deepcopy(matrix)
        matrix.data = np.ones_like(matrix.data)
        item_cooccurrence = matrix.transpose().dot(matrix)
        item_cooccurrence = item_cooccurrence.multiply(item_cooccurrence >= self._threshold)
        return item_cooccurrence.astype(np.float32)

    def train(self, train: sparse.csr_matrix) -> None:
        self._train_matrix = deepcopy(train)
        item_cooccurrence = self._compute_cooccurrence_matrix(train)
        self._item_frequencies = item_cooccurrence.diagonal()
        self._item_similarity = self._similarity_func(item_cooccurrence).astype(np.float32)

    def fit(
        self,
        train: sparse.csr_matrix,
        valid: sparse.csr_matrix = None,
        config: Dict[str, Any] = None,
    ) -> None:
        # Train part
        self.train(train)
        # Validation part
        if valid is not None:
            scores = self.predict(train.toarray())
            target = np.not_equal(valid.toarray(), 0).astype(np.float32)
            for metric in self._metrics:
                metric(scores, target)
            return self.get_metrics(reset=True)

    def predict(self, data: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        scores = self._train_matrix.dot(self._item_similarity)
        if isinstance(scores, sparse.spmatrix):
            scores = scores.toarray()
        if remove_seen:
            scores[data > 0] = -1e13
        return scores

    def get_metrics(self, reset: bool = False) -> None:
        metrics = {}
        for metric in (metric.get_metric(reset) for metric in self._metrics):
            metrics.update(metric)
        return metrics


if __name__ == "__main__":
    model = SAR()
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit(train=data, valid=data)
    print(metrics)
