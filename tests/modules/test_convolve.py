import unittest

import numpy as np
import matplotlib.pyplot as plt

from modules.convolve import Convolve


class TestConvolve(unittest.TestCase):
    def setUp(self):
        self.average = 30

        x = np.linspace(0, 10, 100)
        self.target = 5 + np.sin(x) + np.random.randn(100) * 0.2

    def test_convolve(self):
        convolve = Convolve(self.average, self.target)
        c = convolve.calculate()

        self.assertEqual(type(np.ndarray(1)), type(c))

        plt.plot(self.target, label="origin")
        plt.plot(c, label="convolve")
        plt.legend()
        plt.savefig("test_convolve.png")
