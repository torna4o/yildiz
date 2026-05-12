from .base import DataSource
import numpy as np


class LocalCSV(DataSource):

    def __init__(self, path):
        self.path = path

    def load(self):

        data = np.loadtxt(
            self.path,
            delimiter=","
        )

        return {
            "t": data[:, 0],
            "y": data[:, 1],
            "meta": {
                "source": "local_csv",
                "path": self.path
            }
        }
