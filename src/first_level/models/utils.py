import numpy as np
import pandas as pd
from scipy import sparse


def exponential_decay(value: np.ndarray, max_val: float, half_life: float) -> np.ndarray:
    """
    Compute decay factor for a given value based on an exponential decay.
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
    """
    Helper method to calculate the Jaccard similarity of a matrix of co-occurrences.
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
    """
    Helper method to calculate the Lift of a matrix of co-occurrences. In comparison
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


def apply_time_decay(
    df: pd.DataFrame,
    decay_col: str,
    col_user: str = "user",
    col_item: str = "item",
    col_timestamp: str = "timestamp",
    time_decay_coefficient: float = 30.0,
) -> pd.DataFrame:
    time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
    time_now = df[col_timestamp].max()
    # apply time decay to each rating
    df[decay_col] *= exponential_decay(
        value=df[col_timestamp].values,
        max_val=time_now,
        half_life=time_decay_half_life,
    )
    # group time decayed ratings by user-item and take the sum as the user-item affinity
    return df.groupby([col_user, col_item]).sum().reset_index()
