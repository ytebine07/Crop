import numpy as np
import matplotlib.pyplot as plt


def main():
    common = Common()

    file_path = "./positions.txt"
    x, y = common.get_x_y(file_path)

    xx, yy = common.get_x_y(file_path)
    conv = Convolve(60, yy)
    yy = conv.calculate()

    xxx, yyy = common.get_x_y(file_path)
    conv = Convolve(30, yyy)
    yyy = conv.calculate()

    plt.figure(figsize=(20, 5), dpi=100)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40,
                45, 50, 55, 60, 65, 70, 75, 80, 85, 90])

    plt.xlabel("seconds")
    plt.ylabel("center-position")

    plt.grid(axis='x', which="major")

    plt.plot(x, y, label='original', color="blue")
    plt.plot(xx, yy, label='average-60', color="orange")
    plt.plot(xxx, yyy, label='average-30', color="green")

    plt.legend()
    plt.savefig("graph.png")


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from utils import Convolve, Common
    main()
