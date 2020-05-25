import numpy as np


class Convolve:
    def __init__(self, average: int, target: np.ndarray):
        super().__init__()
        self._average = average
        self._target = target

    def calculate(self):
        mean = self._target.mean()
        offseted_input = self._target - mean

        v = np.ones(self._average) / float(self._average)
        y3 = np.convolve(offseted_input, v, mode="same")

        return y3 + mean
