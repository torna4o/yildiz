# yildiz/preprocessing.py


class Preprocessor:
    """
    Base interface for all preprocessing algorithms.
    """

    def run(self, lightcurve):
        """
        Execute preprocessing on a standard lightcurve dict.

        Parameters
        ----------
        lightcurve : dict
            Must contain:
            {
                "t": array,
                "y": array,
                "meta": dict (optional)
            }

        Returns
        -------
        dict
            Standardized output with "meta".
        """
        raise NotImplementedError