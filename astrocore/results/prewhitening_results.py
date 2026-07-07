

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class PrewhiteningResult:
    """
    Stores the results of a pre-whitening fit.

    Attributes
    ----------
    time : np.ndarray
        Observation times.

    observations : np.ndarray
        Original observations.

    model : np.ndarray
        Reconstructed model evaluated at every observation.

    residuals : np.ndarray
        Observations after subtraction of the fitted model.

    frequencies : np.ndarray
        Frequencies included in the simultaneous fit.

    harmonics : int
        Number of harmonics fitted for every frequency.

    offset : float
        Constant offset of the fitted model.

    amplitudes : np.ndarray
        Amplitudes of every fitted sinusoidal component.

    phases : np.ndarray
        Phases of every fitted sinusoidal component in radians.

    coefficients : np.ndarray
        Raw least-squares coefficients returned by NumPy.

    rank : int
        Rank of the design matrix.

    residual_sum_of_squares : float
        Residual sum of squares reported by NumPy.

    performance : dict | None
        Performance metrics of the pre-whitening fit.
    """

    time: np.ndarray
    observations: np.ndarray

    model: np.ndarray
    residuals: np.ndarray

    frequencies: np.ndarray
    harmonics: int

    offset: float

    amplitudes: np.ndarray
    phases: np.ndarray

    coefficients: np.ndarray

    rank: int
    residual_sum_of_squares: float
    performance: dict | None = field(default=None, compare=False)