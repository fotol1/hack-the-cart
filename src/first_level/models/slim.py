from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from scipy import sparse
from src.metrics import NDCG, Recall
from sklearn.linear_model import ElasticNet
from src.first_level.models.core import Model


class SLIM(Model):
    def __init__(
        self,
        elastic_net: ElasticNet,
        local_topk: int = 100,
        metrics_topk: List[int] = [20, 50, 100],
    ) -> None:
        self._elastic_net = elastic_net
        self._local_topk = local_topk
        self._metrics = [
            NDCG(metrics_topk),
            Recall(metrics_topk),
        ]

    def _train(self, train: sparse.csr_matrix) -> None:
        train = sparse.csc_matrix(train)
        n_items = train.shape[1]
        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10_000_000
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)
        num_cells = 0
        # fit each item's factors sequentially (not in parallel)
        for current_item in tqdm(range(n_items), desc="Train SLIM", total=n_items):
            # get the target column
            y = train[:, current_item].toarray()
            # set the j-th column of X to zero
            start_pos = train.indptr[current_item]
            end_pos = train.indptr[current_item + 1]
            current_item_data_backup = train.data[start_pos:end_pos].copy()
            train.data[start_pos:end_pos] = 0.0
            # fit one ElasticNet model per column
            self._elastic_net.fit(train, y)
            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            nonzero_model_coef_index = self._elastic_net.sparse_coef_.indices
            nonzero_model_coef_value = self._elastic_net.sparse_coef_.data
            local_topk = min(len(nonzero_model_coef_value) - 1, self._local_topk)
            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topk)[
                0:local_topk
            ]
            relevant_items_partition_sorting = np.argsort(
                -nonzero_model_coef_value[relevant_items_partition]
            )
            ranking = relevant_items_partition[relevant_items_partition_sorting]
            for index in range(len(ranking)):
                if num_cells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))
                rows[num_cells] = nonzero_model_coef_index[ranking[index]]
                cols[num_cells] = current_item
                values[num_cells] = nonzero_model_coef_value[ranking[index]]
                num_cells += 1
            # finally, replace the original values of the j-th column
            train.data[start_pos:end_pos] = current_item_data_backup
        # generate the sparse weight matrix
        self._item_matrix = sparse.csr_matrix(
            (values[:num_cells], (rows[:num_cells], cols[:num_cells])),
            shape=(n_items, n_items),
            dtype=np.float32,
        ).toarray()

    def fit(
        self,
        train: sparse.csr_matrix,
        valid: sparse.csr_matrix = None,
        config: Dict[str, Any] = None,
    ) -> None:
        self._train(train)
        if valid is not None:
            scores = self.predict(train.toarray())
            target = np.not_equal(valid.toarray(), 0).astype(np.float32)
            for metric in self._metrics:
                metric(scores, target)
            return self.get_metrics(reset=True)

    def predict(self, data: np.ndarray, remove_seen: bool = True) -> np.ndarray:
        scores = data.dot(self._item_matrix)
        if remove_seen:
            scores[data > 0] = -1e13
        return scores

    def get_metrics(self, reset: bool = False) -> None:
        metrics = {}
        for metric in (metric.get_metric(reset) for metric in self._metrics):
            metrics.update(metric)
        return metrics


if __name__ == "__main__":
    model = SLIM(
        elastic_net=ElasticNet(
            alpha=1.0,
            l1_ratio=0.1,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection="random",
            max_iter=100,
            tol=1e-4,
        ),
    )
    data = sparse.csr_matrix(np.random.randint(low=0, high=2, size=(100, 300)))
    metrics = model.fit(train=data, valid=data)
    print(metrics)
