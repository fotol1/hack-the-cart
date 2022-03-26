from typing import Dict, Union
import pandas as pd
from pathlib import Path
from scipy import sparse
from dataclasses import dataclass


@dataclass
class FitResult:
    metrics: Dict[str, float]
    predicts: pd.DataFrame


class Model:
    def fit(
        self,
        train: Union[Path, sparse.csr_matrix],
        valid: Union[Path, sparse.csr_matrix],
    ) -> Dict[str, float]:
        raise NotImplementedError()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError()

    def save(serialization_dir: Path) -> None:
        raise NotImplementedError()

    @classmethod
    def load(serialization_dir: Path) -> "Model":
        raise NotImplementedError()
