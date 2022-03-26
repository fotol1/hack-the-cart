from typing import List, Dict, Any
import numpy as np
from scipy import sparse
from src.metrics import NDCG, Recall
from sklearn.preprocessing import normalize
from src.first_level.models.core import Model


class EASE(Model):
    def __init__(
        self,
        norm: bool = True,
        reg_weight: float = 100.0,
        metrics_topk: List[int] = [20, 50, 100],
    ) -> None:
        self._norm = norm
        self._reg_weight = reg_weight
        self._item_matrix = None
        self._metrics = [
            NDCG(metrics_topk),
            Recall(metrics_topk),
        ]

    def fit(
        self,
        config: Dict[str, Any],
        train: np.ndarray,
        valid: np.ndarray = None,
    ) -> None:
        # Train part
        X = train
        if self._norm:
            X = normalize(X, norm="l2", axis=1)
            X = normalize(X, norm="l2", axis=0)
            X = sparse.csr_matrix(X)
        # gram matrix
        G = X.T @ X
        # add reg to diagonal
        G += self._reg_weight * sparse.identity(G.shape[0])
        # convert to dense because inverse will be dense
        G = G.todense()
        # invert. this takes most of the time
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        # zero out diag
        np.fill_diagonal(B, 0.0)
        self._item_matrix = B
        # Validation part
        prediction_scores = self.predict(train.toarray())
        for metric in self._metrics:
            metric(prediction_scores, valid.toarray())
        return self.get_metrics(reset=True)

    def predict(self, data: np.ndarray) -> np.ndarray:
        scores = data.dot(self._item_matrix)
        scores[data > 0] = -1e13
        return scores

    def get_metrics(self, reset: bool = False) -> None:
        metrics = {}
        for metric in (metric.get_metric(reset) for metric in self._metrics):
            metrics.update(metric)
        return metrics


if __name__ == "__main__":
    model = EASE()
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit({}, train=data, valid=data)
    print(metrics)
