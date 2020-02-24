import numpy as np
import matplotlib.pyplot as plt


def main():
    common = Common()

    file_path = "./positions_normal.txt"
    x, y = common.get_x_y(file_path)

    xx, yy = common.get_x_y(file_path)
    conv = Convolve(60, yy)
    yy = conv.calculate()

    xxx, yyy = common.get_x_y(file_path)
    conv = Convolve(30, yyy)
    yyy = conv.calculate()

    plt.figure(figsize=(20, 5), dpi=100)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    plt.xlabel("seconds")
    plt.ylabel("center-position")

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
