"""
Sinusoidal pre-whitening.

This module provides simultaneous least-squares fitting of one or more
frequencies and their harmonics using NumPy only.
"""

from __future__ import annotations

import numpy as np

from yildiz.results.prewhitening_results import PrewhiteningResult
from yildiz.preprocessing import Preprocessor
from yildiz.preprocessors.fft import FFT

def _build_design_matrix(
    time: np.ndarray,
    frequencies: np.ndarray,
    harmonics: int,
) -> np.ndarray:
    """
    Construct the least-squares design matrix.

    Parameters
    ----------
    time : np.ndarray
        Observation times.

    frequencies : np.ndarray
        Frequencies to fit.

    harmonics : int
        Number of harmonics to include. A value of 1 fits only the
        fundamental frequency.

    Returns
    -------
    np.ndarray
        Design matrix.
    """
    columns = [np.ones_like(time)]

    for frequency in frequencies:
        for harmonic in range(1, harmonics + 1):
            omega = 2.0 * np.pi * harmonic * frequency

            columns.append(np.cos(omega * time))
            columns.append(np.sin(omega * time))

    return np.column_stack(columns)

def _evaluate_model(
    design_matrix: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the fitted model.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix.

    coefficients : np.ndarray
        Least-squares coefficients.

    Returns
    -------
    np.ndarray
        Evaluated model.
    """
    return design_matrix @ coefficients

def _extract_harmonic_parameters(
    coefficients: np.ndarray,
    n_frequencies: int,
    harmonics: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract harmonic amplitudes and phases from least-squares
    coefficients.

    Parameters
    ----------
    coefficients : np.ndarray
        Least-squares solution vector.

    n_frequencies : int
        Number of fitted fundamental frequencies.

    harmonics : int
        Number of harmonics fitted for each frequency.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Harmonic amplitudes and phases.
    """
    amplitudes = []
    phases = []

    index = 1

    for _ in range(n_frequencies):

        for _ in range(harmonics):

            cosine = coefficients[index]
            sine = coefficients[index + 1]

            amplitudes.append(
                np.hypot(cosine, sine)
            )

            phases.append(
                np.arctan2(
                    -sine,
                    cosine,
                )
            )

            index += 2

    return (
        np.asarray(amplitudes),
        np.asarray(phases),
    )


def _find_peaks(
    amplitudes: np.ndarray,
) -> np.ndarray:
    """
    Find local maxima in an amplitude spectrum.

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitude spectrum.

    Returns
    -------
    np.ndarray
        Indices of local maxima.
    """
    if amplitudes.size < 3:
        return np.array([], dtype=int)

    peaks = []

    for index in range(1, amplitudes.size - 1):

        if (
            amplitudes[index] > amplitudes[index - 1]
            and amplitudes[index] >= amplitudes[index + 1]
        ):
            peaks.append(index)

    return np.asarray(peaks, dtype=int)

def _estimate_local_noise(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    center_frequency: float,
    window_width: float = 1.0,
    exclusion_width: float = 0.05,
) -> float:
    """
    Estimate the local noise level around a frequency.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis.

    amplitudes : np.ndarray
        Amplitude spectrum.

    center_frequency : float
        Frequency around which the noise is estimated.

    window_width : float, optional
        Half-width of the local frequency window.

    exclusion_width : float, optional
        Half-width excluded around the signal frequency.

    Returns
    -------
    float
        Median local amplitude.
    """
    window_mask = (
        np.abs(frequencies - center_frequency)
        <= window_width
    )

    exclusion_mask = (
        np.abs(frequencies - center_frequency)
        <= exclusion_width
    )

    noise_mask = (
        window_mask
        & ~exclusion_mask
    )

    if np.count_nonzero(noise_mask) == 0:
        return 0.0

    return np.median(
        amplitudes[noise_mask]
    )

def _mask_peak_region(
    candidate_indices: np.ndarray,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    exclusion_width: float,
) -> np.ndarray:
    """
    Remove candidate peaks that lie within the exclusion width
    of a stronger peak.

    Parameters
    ----------
    candidate_indices : np.ndarray
        Indices of detected local maxima.

    frequencies : np.ndarray
        Frequency axis.

    amplitudes : np.ndarray
        Amplitude spectrum.

    exclusion_width : float
        Frequency exclusion half-width.

    Returns
    -------
    np.ndarray
        Filtered candidate indices.
    """
    if candidate_indices.size == 0:
        return candidate_indices

    order = np.argsort(
        amplitudes[candidate_indices]
    )[::-1]

    selected = []

    for idx in candidate_indices[order]:

        frequency = frequencies[idx]

        keep = True

        for accepted in selected:

            if (
                abs(
                    frequency
                    - frequencies[accepted]
                )
                < exclusion_width
            ):
                keep = False
                break

        if keep:
            selected.append(idx)

    return np.asarray(selected, dtype=int)

def _fit_frequencies(
    time: np.ndarray,
    observations: np.ndarray,
    frequencies: np.ndarray,
    harmonics: int,
) -> PrewhiteningResult:
    
    time = np.asarray(time, dtype=float)
    observations = np.asarray(observations, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    if time.ndim != 1:
        raise ValueError("'time' must be one-dimensional.")

    if observations.ndim != 1:
        raise ValueError("'observations' must be one-dimensional.")

    if frequencies.ndim != 1:
        raise ValueError("'frequencies' must be one-dimensional.")

    if time.size != observations.size:
        raise ValueError(
            "'time' and 'observations' must have the same length."
        )

    if frequencies.size == 0:
        raise ValueError(
            "At least one frequency must be provided."
        )

    if harmonics < 1:
        raise ValueError(
            "'harmonics' must be at least 1."
        )


    design_matrix = _build_design_matrix(
        time=time,
        frequencies=frequencies,
        harmonics=harmonics,
    )

    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design_matrix,
        observations,
        rcond=None,
    )

    model = _evaluate_model(design_matrix, coefficients)
    residual_vector = observations - model

    offset = coefficients[0]

    amplitudes, phases = _extract_harmonic_parameters(
        coefficients=coefficients,
        n_frequencies=len(frequencies),
        harmonics=harmonics,
    )


    if residuals.size:
        residual_sum_of_squares = float(residuals[0])
    else:
        residual_sum_of_squares = float(
            np.sum(residual_vector ** 2)
        )

    return PrewhiteningResult(
        time=time,
        observations=observations,
        model=model,
        residuals=residual_vector,
        frequencies=frequencies,
        harmonics=harmonics,
        offset=offset,
        amplitudes=amplitudes,
        phases=phases,
        coefficients=coefficients,
        rank=rank,
        residual_sum_of_squares=residual_sum_of_squares,
    )




def prewhiten(
    time: np.ndarray,
    observations: np.ndarray,
    peak_indices: np.ndarray | None = None,
    top_n: int | None = None,
    snr_threshold: float | None = None,
    snr_window_width: float = 1.0,
    snr_exclusion_width: float = 0.05,
    harmonics: int = 1,
    preprocessor: Preprocessor | None = None,
) -> PrewhiteningResult:
    """
    Perform simultaneous sinusoidal pre-whitening.

    Parameters
    ----------
    time : np.ndarray
        Observation times.

    observations : np.ndarray
        Observed values.

    frequencies : np.ndarray
        Frequencies to fit.

    harmonics : int, optional
        Number of harmonics to fit for each frequency.
        Default is 1 (fundamental only).

    Returns
    -------
    PrewhiteningResult
        Result of the pre-whitening fit.

    Raises
    ------
    ValueError
        If the input arrays have incompatible dimensions or invalid values.
    """
    time = np.asarray(time, dtype=float)
    observations = np.asarray(observations, dtype=float)

    if time.ndim != 1:
        raise ValueError("'time' must be one-dimensional.")

    if observations.ndim != 1:
        raise ValueError("'observations' must be one-dimensional.")


    if time.size != observations.size:
        raise ValueError(
            "'time' and 'observations' must have the same length."
        )

    if harmonics < 1:
        raise ValueError("'harmonics' must be at least 1.")

    accepted_frequencies = []

    residual = observations.copy()
    if preprocessor is None:
        preprocessor = FFT(
            detrend=True,
            remove_dc=True,
        )
    while True:

        spectrum = preprocessor.run(
            {
                "t": time,
                "y": residual,
            }
        )

        amplitudes = spectrum.amplitudes

        candidate_indices = _find_peaks(
            amplitudes,
        )

        candidate_indices = _mask_peak_region(
            candidate_indices=candidate_indices,
            frequencies=spectrum.frequencies,
            amplitudes=amplitudes,
            exclusion_width=0.05,
        )

        if candidate_indices.size == 0:
            break

        peak_indices = candidate_indices[
            np.argsort(
                amplitudes[candidate_indices]
            )[::-1]
        ]

        index = peak_indices[0]

        frequency = spectrum.frequencies[index]

        signal = amplitudes[index]

        noise = _estimate_local_noise(
            frequencies=spectrum.frequencies,
            amplitudes=amplitudes,
            center_frequency=frequency,
            window_width=snr_window_width,
            exclusion_width=snr_exclusion_width,
        )

        if noise == 0.0:
            break

        snr = signal / noise

        if (
            snr_threshold is not None
            and snr < snr_threshold
        ):
            break

        accepted_frequencies.append(frequency)

        if (
            top_n is not None
            and len(accepted_frequencies) >= top_n
        ):
            break

        result = _fit_frequencies(
            time=time,
            observations=observations,
            frequencies=np.asarray(
                accepted_frequencies,
                dtype=float,
            ),
            harmonics=harmonics,
        )

        residual = result.residuals

    frequencies = np.asarray(
        accepted_frequencies,
        dtype=float,
    )

    if frequencies.size == 0:
        raise ValueError(
            "No frequencies satisfied the stopping criteria."
        )

    return _fit_frequencies(
        time=time,
        observations=observations,
        frequencies=frequencies,
        harmonics=harmonics,
    )
    