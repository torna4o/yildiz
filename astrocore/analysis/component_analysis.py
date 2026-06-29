# astrocore/analysis/component_analysis.py

from __future__ import annotations

import numpy as np

from astrocore.preprocessors.fft import FFT
from astrocore.preprocessors.ssa import SSAResult


class ComponentAnalyzer:
    """
    Central analysis engine for SSA components.

    Responsibility
    ---------------
    Takes SSAResult and attaches analysis products
    (FFT, future spectral tools, statistics, etc.)
    to each SSAComponent.

    IMPORTANT:
    - Does NOT perform decomposition
    - Does NOT modify SSA logic
    """

    # --------------------------------------------------------
    # ANALYSIS REGISTRY
    # --------------------------------------------------------

    def __init__(self):

        self._registry = {
            "fft": self._fft_analysis,
        }

    # --------------------------------------------------------

    def analyze(
        self,
        result: SSAResult,
        analyses: list[str] | None = None
    ) -> SSAResult:
        """
        Run selected analyses on SSA components.

        Parameters
        ----------
        result : SSAResult
            Output of SSA decomposition.

        analyses : list[str]
            Supported:
                - "fft"

        Returns
        -------
        SSAResult (mutated in-place)
        """

        if analyses is None:
            return result

        for name in analyses:

            if name not in self._registry:

                raise ValueError(
                    f"Unknown analysis: {name}"
                )

        for component in result.components:

            for name in analyses:

                self._registry[name](component)

        return result

    # --------------------------------------------------------

    def _fft_analysis(self, component) -> None:
        """
        Attach FFT analysis to SSA component.
        """

        fft = FFT()

        output = fft.run(
            {
                "t": component.time,
                "y": component.flux
            }
        )

        component.analyses["fft"] = output
