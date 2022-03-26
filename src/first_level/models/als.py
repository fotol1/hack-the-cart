from typing import Dict, List, Union, Tuple, Any
import numpy as np
from tqdm import tqdm
from scipy import sparse
from pathlib import Path
from copy import deepcopy
from itertools import islice
from src.metrics import NDCG, Recall
from scipy.sparse.linalg import spsolve
from implicit.als import AlternatingLeastSquares
from src.first_level.models.core import Model, FitResult


class ALS(Model):
    def __init__(
        self,
        inner_model: AlternatingLeastSquares,
        topk: int = 200,
        metrics_topk: List[int] = [20, 100],
        batch_size: int = 128,
    ) -> None:
        self.inner_model = inner_model
        self._topk = topk
        self._batch_size = batch_size
        self._metrics = [
            NDCG(metrics_topk),
            Recall(metrics_topk),
        ]

    def fit(
        self,
        config: Dict[str, Any],
        train: Union[Path, sparse.csr_matrix],
        valid: Union[Path, sparse.csr_matrix],
    ) -> FitResult:
        """Fit iALS on sparse matrix of user to item interactions."""
        # Force data to be sparse csr matrix just in case.
        if isinstance(train, Path):
            train = sparse.load_npz(train)
        if isinstance(valid, Path):
            valid = sparse.load_npz(valid)
        self.inner_model.fit(train, show_progress=True)
        return self.validate(valid)

    def validate(self, data: sparse.csr_matrix) -> FitResult:
        # Compute metrics by iterating over interaction matrix.
        iterator = range(data.shape[0])
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
            output = self.predict(users, with_fit=False)
            target = np.not_equal(data[start:stop].toarray(), 0).astype(np.float32)[:, : self._topk]
            for metric in self._metrics:
                metric(output, target)
        # Process leftovers if needed.
        if stop < data.shape[0]:
            users = np.arange(stop, data.shape[0])
            output = self.predict(users, with_fit=False)
            target = np.not_equal(data[stop:].toarray(), 0).astype(np.float32)[:, : self._topk]
            for metric in self._metrics:
                metric(output, target)
        # user_preds_df = pd.DataFrame(
        #     {
        #         "user": all_users,
        #         "item": [user_preds[user][0] for user in all_users],
        #         "score": [user_preds[user][1] for user in all_users],
        #     }
        # ).explode(["item", "score"])
        return self.get_metrics(reset=True)

    def predict(self, users: np.ndarray = None, with_fit: bool = True) -> np.ndarray:
        """Get prediction only for certain users. If None perform computation for all the users."""
        if with_fit:
            return self.predict_with_fit(users)
        user_factors = (
            self.inner_model.user_factors[users]
            if users is not None
            else self.inner_model.user_factors
        )
        return np.einsum("uh,ih->ui", user_factors, self.inner_model.item_factors)[:, : self._topk]

    def recommend(self, users: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.predict(users, with_fit=False)
        sorted_indices = np.argsort(-scores, axis=-1)
        sorted_scores = np.sort(-scores, axis=-1)
        return sorted_scores, sorted_indices

    def predict_with_fit(self, user_vectors: np.ndarray) -> np.ndarray:
        lambda_val = 0.1
        item_factors = self.inner_model.item_factors
        yTy = item_factors.T.dot(item_factors)  # d * d matrix

        # X = np.random.normal(size=(1, item_factors.shape[1]))
        Y_eye = sparse.eye(item_factors.shape[0])

        lambda_eye = lambda_val * np.ones(
            (item_factors.shape[1], item_factors.shape[1])
        )  # * sparse.eye(item_factors.shape[1])

        # Compute yTy and xTx at beginning of each iteration to save computing time

        result_vectors = []
        for user_vector in user_vectors:
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
        print(len(result_vectors), result_vectors[0].shape)
        return np.array(result_vectors)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric in (metric.get_metric(reset) for metric in self._metrics):
            metrics.update(metric)
        return metrics


if __name__ == "__main__":
    model = ALS(
        inner_model=AlternatingLeastSquares(factors=64, num_threads=4, iterations=100),
        batch_size=32,
    )
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit({}, train=data, valid=data)
    check = model.predict(data.toarray(), with_fit=True)
    print(check)
    print(check.shape)
