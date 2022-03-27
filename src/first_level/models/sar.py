from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse
from copy import deepcopy
from src.metrics import NDCG, Recall
from src.first_level.models.core import Model


def exponential_decay(value, max_val, half_life):
    """Compute decay factor for a given value based on an exponential decay.
    Values greater than `max_val` will be set to 1.
    Args:
        value (numeric): Value to calculate decay factor
        max_val (numeric): Value at which decay factor will be 1
        half_life (numeric): Value at which decay factor will be 0.5
    Returns:
        float: Decay factor
    """
    return np.minimum(1.0, np.power(0.5, (max_val - value) / half_life))


def jaccard(cooccurrence: sparse.csr_matrix) -> np.ndarray:
    """Helper method to calculate the Jaccard similarity of a matrix of co-occurrences.
    When comparing Jaccard with count co-occurrence and lift similarity, count favours
    predictability, meaning that the most popular items will be recommended most of
    the time. Lift, by contrast, favours discoverability/serendipity, meaning that an
    item that is less popular overall but highly favoured by a small subset of users
    is more likely to be recommended. Jaccard is a compromise between the two.
    Args:
        cooccurrence (numpy.ndarray): the symmetric matrix of co-occurrences of items.
    Returns:
        numpy.ndarray: The matrix of Jaccard similarities between any two items.
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows + diag_cols - cooccurrence)

    return np.array(result)


def lift(cooccurrence: sparse.csr_matrix) -> np.ndarray:
    """Helper method to calculate the Lift of a matrix of co-occurrences. In comparison
    with basic co-occurrence and Jaccard similarity, lift favours discoverability and
    serendipity, as opposed to co-occurrence that favours the most popular items, and
    Jaccard that is a compromise between the two.
    Args:
        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.
    Returns:
        numpy.ndarray: The matrix of Lifts between any two items.
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows * diag_cols)

    return np.array(result)


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
        time_decay_coefficient: float = 30,
        metrics_topk: List[int] = [20, 50, 100],
    ) -> None:
        assert similarity in ("jaccard", "lift", "cooccurrence")
        self._threshold = threshold
        self._train_matrix = None
        self._item_similarity = None
        self._item_frequencies = None
        self._time_decaty_coefficient = time_decay_coefficient
        self._time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
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

    def apply_time_decay(
        self,
        df: pd.DataFrame,
        decay_col: str,
        col_user: str = "user",
        col_item: str = "item",
        col_timestamp: str = "timestamp",
    ) -> pd.DataFrame:
        self._time_now = df[col_timestamp].max()
        # apply time decay to each rating
        df[decay_col] *= exponential_decay(
            value=df[col_timestamp],
            max_val=self._time_now,
            half_life=self._time_decay_half_life,
        )
        # group time decayed ratings by user-item and take the sum as the user-item affinity
        return df.groupby([self.col_user, self.col_item]).sum().reset_index()


if __name__ == "__main__":
    model = SAR()
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit(train=data, valid=data)
    print(metrics)
