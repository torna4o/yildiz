import numpy as np
from astrocore.preprocessing import Preprocessor
from astrocore.utils.performance import measure_performance


class FFT(Preprocessor):
    """
    FFT spectral decomposition.

    KEY CONCEPTS (read these if unclear):
    - Power spectrum: https://en.wikipedia.org/wiki/Spectral_density
    - Signal energy (DSP): https://en.wikipedia.org/wiki/Parseval%27s_theorem
    - Period-frequency relation: https://en.wikipedia.org/wiki/Frequency

    NOTE:
    In this context, "energy" = squared magnitude of Fourier coefficients,
    not physical energy unless the signal is calibrated.
    """

    def __init__(self, detrend=True, remove_dc=True):
        self.detrend = detrend
        self.remove_dc = remove_dc

    @measure_performance
    def run(self, lightcurve):

        t = lightcurve["t"]
        y = lightcurve["y"]

        if len(t) < 2:
            raise ValueError("Time series too short for FFT.")

        dt = np.median(np.diff(t))

        y = np.asarray(y)

        if self.detrend:
            y = y - np.nanmean(y)

        freq = np.fft.rfftfreq(len(y), d=dt)
        fft_vals = np.fft.rfft(y)

        power = np.abs(fft_vals) ** 2

        if self.remove_dc:
            mask = freq > 0
        else:
            mask = np.ones_like(freq, dtype=bool)

        freq = freq[mask]
        power = power[mask]

        period = 1.0 / freq
        omega = 2.0 * np.pi * freq

        return {
            "frequency": freq,
            "period": period,
            "angular_frequency": omega,
            "power": power,

            "meta": {
                "method": "FFT",
                "cadence": dt,
                "n_points": len(y),
                "detrend": self.detrend,
                "remove_dc": self.remove_dc
            }
        }
