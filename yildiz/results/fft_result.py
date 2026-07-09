# Module to store FFT computation results in a structured way.

from dataclasses import dataclass, field

import numpy as np

@dataclass(frozen=True, slots=True)
class FFTResult:
    """
    Stores the results of a Fast Fourier Transform.

    Attributes
    ----------
    time : np.ndarray
        Observation times.

    observations : np.ndarray
        Original observations.

    frequencies : np.ndarray
        Frequency axis.

    amplitudes : np.ndarray
        Amplitude spectrum.

    phases : np.ndarray
        Phase spectrum in radians.

    fft : np.ndarray
        Complex FFT coefficients.

    sampling_interval : float
        Median sampling interval.

    sampling_frequency : float
        Sampling frequency.

    nyquist_frequency : float
        Nyquist frequency.
    
    performance : dict | None
        Performance metrics of the pre-whitening fit.
    """

    time: np.ndarray
    observations: np.ndarray

    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray

    fft: np.ndarray

    sampling_interval: float
    sampling_frequency: float
    nyquist_frequency: float
    performance: dict | None = field(default=None, compare=False)