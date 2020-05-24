import numpy as np


class Convolve:
    def __init__(self, average: int, input: np.ndarray):
        """
        @args
            average : 移動平均を求めるフレーム数
            input   : ポジションのlist
        @see
            https://deepage.net/features/numpy-convolve.html
        @return
            移動平均後の整数の座標のlist
        """
        self._average = average
        self._input = input

    def calculate(self) -> np.ndarray:
        mean = self._input.mean()
        offseted_input = self._input - mean

        v = np.ones(self._average) / float(self._average)
        y3 = np.convolve(offseted_input, v, mode="same")

        return y3 + mean
