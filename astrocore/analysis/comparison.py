# yildiz/analysis/comparison.py

import numpy as np


class TimeSeriesComparator:
    """
    Comparison engine for SSA-based decomposition outputs.

    This version operates on:
    - SSAComponent objects
    - Component-level FFT results

    It does NOT use:
    - eigenvalues
    - raw SSA matrices
    - chunk-level heuristics

    It compares STRUCTURE in spectral space.
    """

    # =========================================================
    # FFT-BASED COMPONENT COMPARISON
    # =========================================================

    @staticmethod
    def compare_component_fft(comp_a, comp_b):
        """
        Compare FFT structure of two SSA components.

        Uses:
        - cosine similarity of normalized power spectra
        """

        fft_a = comp_a.analyses.get("fft", None)
        fft_b = comp_b.analyses.get("fft", None)

        if fft_a is None or fft_b is None:
            raise ValueError("FFT analysis missing in one or both components.")

        p1 = np.asarray(fft_a["power"])
        p2 = np.asarray(fft_b["power"])

        # normalize (removes amplitude scaling bias)
        p1 = p1 / (np.linalg.norm(p1) + 1e-12)
        p2 = p2 / (np.linalg.norm(p2) + 1e-12)

        similarity = float(np.dot(p1, p2))

        return {
            "similarity": similarity,
            "metric": "component_fft_cosine_similarity"
        }

    # =========================================================
    # SSA COMPONENT SET COMPARISON (ENERGY STRUCTURE ONLY)
    # =========================================================

    @staticmethod
    def compare_energy_profiles(ssa_a, ssa_b):
        """
        Compare energy distribution across SSA components.

        Important:
        - this is NOT eigenvalue comparison anymore
        - uses SSAComponent.energy_fraction

        Interpretation:
        - compares how variance is distributed across modes
        """

        e1 = np.array([c.energy_fraction for c in ssa_a.components])
        e2 = np.array([c.energy_fraction for c in ssa_b.components])

        # normalize safely
        e1 = e1 / (np.sum(e1) + 1e-12)
        e2 = e2 / (np.sum(e2) + 1e-12)

        m = min(len(e1), len(e2))

        e1 = e1[:m]
        e2 = e2[:m]

        similarity = float(
            np.dot(e1, e2) /
            (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-12)
        )

        return {
            "similarity": similarity,
            "metric": "ssa_component_energy_profile_similarity"
        }

    # =========================================================
    # RAW CHUNK COMPARISON (STILL USEFUL, BUT LOW LEVEL)
    # =========================================================

    @staticmethod
    def compare_chunks(chunk_a, chunk_b):
        """
        Compare raw time-series chunks (pre-SSA stage).
        """

        y1 = np.asarray(chunk_a["y"])
        y2 = np.asarray(chunk_b["y"])

        return {
            "stats": {
                "mean_diff": float(abs(np.mean(y1) - np.mean(y2))),
                "std_ratio": float(np.std(y1) / (np.std(y2) + 1e-12)),
                "length_ratio": float(len(y1) / len(y2)),
            },
            "metric": "raw_chunk_statistics"
        }