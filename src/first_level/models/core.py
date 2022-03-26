from typing import Dict, Union
import numpy as np
from pathlib import Path
from scipy import sparse
from dataclasses import dataclass


class Model:
    def fit(
        self,
        train: Union[Path, sparse.csr_matrix],
        valid: Union[Path, sparse.csr_matrix],
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def predict(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def save(serialization_dir: Path) -> None:
        raise NotImplementedError()

    @classmethod
    def load(serialization_dir: Path) -> "Model":
        raise NotImplementedError()
