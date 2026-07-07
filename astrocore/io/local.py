from .base import DataSource

import numpy as np
from astropy.io import fits


class LocalASCII(DataSource):
    """
    Load an ASCII text table.

    Supports:
    - whitespace-separated columns (spaces or tabs)
    - comma-separated values (CSV)
    - missing values (returned as NaN)

    The first column is interpreted as time and the second as observations.
    """

    def __init__(self, path):
        self.path = path

    def load(
        self,
        header_lines=0,
        delimiter=None,
    ):
        """
        Parameters
        ----------
        header_lines : int, optional
            Number of header lines to skip.

        delimiter : str or None, optional
            Column delimiter.

            None  -> arbitrary whitespace (recommended)
            ","   -> CSV
            "\t"  -> tab-separated
            ";"   -> semicolon-separated
        """

        data = np.genfromtxt(
            self.path,
            delimiter=delimiter,
            skip_header=header_lines,
            dtype=float,
            filling_values=np.nan,
            invalid_raise=False,
        )

        # Remove completely empty rows
        data = data[~np.all(np.isnan(data), axis=1)]

        return {
            "t": data[:, 0],
            "y": data[:, 1],
            "meta": {
                "source": "local_ascii",
                "path": self.path,
                "delimiter": delimiter,
            },
        }

class LocalFITS(DataSource):
    # Class to load lightcurve FITS files downloaded from MAST
    def __init__(self, path):
        self.path = path

    def load(self):

        with fits.open(self.path) as hdul:

            table = hdul[1].data

            t = table["TIME"]
            y = table["PDCSAP_FLUX"]

        mask = np.isfinite(t) & np.isfinite(y)

        return {
            "t": t[mask],
            "y": y[mask],
            "meta": {
                "source": "local_fits",
                "path": self.path
            }
        }
