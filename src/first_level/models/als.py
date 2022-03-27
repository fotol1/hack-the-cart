from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from scipy import sparse
from copy import deepcopy
from itertools import islice
from src.metrics import NDCG, Recall
from scipy.sparse.linalg import spsolve
from src.first_level.models.core import Model
from implicit.als import AlternatingLeastSquares


class Scaling:
    def apply(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        raise NotImplementedError()


class LinearScaling(Scaling):
    def __init__(self, alpha: float = 40.0) -> None:
        self._alpha = alpha

    def apply(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        matrix.data = 1.0 + self._alpha * matrix.data
        return matrix


class LogScaling(Scaling):
    def __init__(self, alpha: float = 1.0, epsilon: float = 1.0) -> None:
        self._alpha = alpha
        self._epsilon = epsilon

    def apply(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        matrix.data = 1.0 + self._alpha * np.log(1.0 + matrix.data / self._epsilon)
        return matrix


class ALS(Model):
    def __init__(
        self,
        inner_model: AlternatingLeastSquares,
        scaling: Scaling = None,
        batch_size: int = 128,
        metrics_topk: List[int] = [20, 50, 100],
    ) -> None:
        self.inner_model = inner_model
        self._scaling = scaling
        self._batch_size = batch_size
        self._metrics = [
            NDCG(metrics_topk),
            Recall(metrics_topk),
        ]
        # Train configs
        self._train_matrix = None

    def fit(
        self,
        train: sparse.csr_matrix,
        valid: sparse.csr_matrix = None,
        config: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Fit iALS on sparse matrix of user to item interactions."""
        self._train_matrix = deepcopy(train)
        if self._scaling is not None:
            self._train_matrix = self._scaling.apply(self._train_matrix)
        self.inner_model.fit(self._train_matrix, show_progress=True)
        if valid is not None:
            return self.validate(valid)

    def validate(self, data: sparse.csr_matrix) -> Dict[str, float]:
        # Compute metrics by iterating over interaction matrix.
        iterator = range(data.shape[0])
        stop = 0
        for start, stop in tqdm(
            # Copy iterators for each islice.
            zip(
                islice(deepcopy(iterator), 0, data.shape[0], self._batch_size),
                islice(deepcopy(iterator), self._batch_size, data.shape[0], self._batch_size),
            ),
            desc="Validating ALS",
            total=data.shape[0] // self._batch_size,
        ):
            users = np.arange(start, stop)
            output = self.get_scores(
                user_idx=users, user_factors=self.inner_model.user_factors[users]
            )
            target = np.not_equal(data[start:stop].toarray(), 0).astype(np.float32)
            for metric in self._metrics:
                metric(output, target)
        # Process leftovers if needed.
        if stop < data.shape[0]:
            users = np.arange(stop, data.shape[0])
            output = self.get_scores(
                user_idx=users, user_factors=self.inner_model.user_factors[users]
            )
            target = np.not_equal(data[stop:].toarray(), 0).astype(np.float32)
            for metric in self._metrics:
                metric(output, target)
        return self.get_metrics(reset=True)

    def get_scores(
        self, user_idx: np.ndarray, user_factors: np.ndarray, remove_seen: bool = True
    ) -> np.ndarray:
        scores = np.einsum("uh,ih->ui", user_factors, self.inner_model.item_factors)
        if remove_seen:
            scores[self._train_matrix[user_idx].toarray() > 0] = -1e13
        return scores

    def predict(self, data: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        lambda_val = 0.1
        item_factors = self.inner_model.item_factors
        yTy = item_factors.T.dot(item_factors)  # d * d matrix
        Y_eye = sparse.eye(item_factors.shape[0])
        lambda_eye = lambda_val * np.ones((item_factors.shape[1], item_factors.shape[1]))
        # Compute yTy and xTx at beginning of each iteration to save computing time
        result_vectors = []
        for user_vector in tqdm(data, desc="Predict with ALS", total=len(data)):
            conf_samp = user_vector.T  # Grab user row from confidence matrix and convert to dense
            pref = conf_samp.copy()
            pref[pref != 0] = 1  # Create binarized preference vector
            CuI = sparse.diags(
                conf_samp, 0
            ).toarray()  # Get Cu - I term, don't need to subtract 1 since we never added it
            # (d, n_items) * (n_items * n_items) * (n_items, d)
            yTCuIY = item_factors.T.dot(CuI).dot(item_factors)  # This is the yT(Cu-I)Y term
            # (d, n_items) * (n_items, n_items) * (n_items, 1)
            yTCupu = item_factors.T.dot(CuI + Y_eye).dot(
                pref.T
            )  # This is the yTCuPu term, where we add the eye back in
            # Cu - I + I = Cu
            xx = sparse.csr_matrix(yTy + yTCuIY + lambda_eye)
            yy = sparse.csr_matrix(yTCupu.T)
            X = spsolve(xx, yy)
            result_vectors.append(X)
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, equation 4 from the paper
            # Begin iteration to solve for Y based on fixed X
        result_vectors = np.array(result_vectors)
        return self.get_scores(
            user_idx=np.arange(result_vectors.shape[0]),
            user_factors=result_vectors,
            remove_seen=remove_seen,
        )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in (metric.get_metric(reset) for metric in self._metrics):
            metrics.update(metric)
        return metrics


if __name__ == "__main__":
    model = ALS(
        inner_model=AlternatingLeastSquares(factors=64, num_threads=4, iterations=100),
        scaling=LinearScaling(),
        batch_size=32,
    )
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit(train=data, valid=data)
    print(metrics)
    check = model.predict(data.toarray())
    print(check)
