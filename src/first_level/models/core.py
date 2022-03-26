from typing import Dict, Any
import numpy as np
from scipy import sparse


class Model:
    def fit(
        self,
        train: sparse.csr_matrix,
        valid: sparse.csr_matrix = None,
        config: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
