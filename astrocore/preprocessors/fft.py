import numpy as np
from yildiz.preprocessing import Preprocessor
from yildiz.results.fft_resul import FFTResult
from yildiz.utils.performance import measure_performance


class FFT(Preprocessor):
    """
    FFT spectral decomposition.

    Notes
    -----
    This implementation assumes uniformly sampled data.
    """

    def __init__(self, detrend=True, remove_dc=True):
        self.detrend = detrend
        self.remove_dc = remove_dc

    @measure_performance
    def run(self, lightcurve):

        time = np.asarray(lightcurve["t"], dtype=float)
        observations = np.asarray(lightcurve["y"], dtype=float)

        if len(time) < 2:
            raise ValueError("Time series too short for FFT.")

        sampling_interval = np.median(np.diff(time))
        sampling_frequency = 1.0 / sampling_interval
        nyquist_frequency = sampling_frequency / 2.0

        if self.detrend:
            observations = observations - np.nanmean(observations)

        frequencies = np.fft.rfftfreq(
            len(observations),
            d=sampling_interval,
        )

        fft = np.fft.rfft(observations)

        amplitudes = np.abs(fft)
        phases = np.angle(fft)
        power = amplitudes ** 2

        if self.remove_dc:
            mask = frequencies > 0.0

            frequencies = frequencies[mask]
            fft = fft[mask]
            amplitudes = amplitudes[mask]
            phases = phases[mask]
            power = power[mask]

        return FFTResult(
            time=time,
            observations=observations,
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            fft=fft,
            sampling_interval=sampling_interval,
            sampling_frequency=sampling_frequency,
            nyquist_frequency=nyquist_frequency,
        )
