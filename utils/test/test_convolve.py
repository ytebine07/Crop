def main():
    path = "/tmp/positions.txt"
    c = Common()
    x, y = c.get_x_y(path)

    target = Convolve(60, y)
    print(target.calculate2())


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    from utils import Convolve, Common
    main()
