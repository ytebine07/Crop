import numpy as np


class Convolve:
    def __init__(self, average: int):
        super().__init__()
        self._average = average

    def calculate(self, target: np.ndarray):
        mean = target.mean()
        offseted_input = target - mean

        v = np.ones(self._average) / float(self._average)
        y3 = np.convolve(offseted_input, v, mode="same")

        return y3 + mean
