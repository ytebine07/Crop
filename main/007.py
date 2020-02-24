def main():
    print("main")
    hoge = Convolve()
    print(hoge.hoge())


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from utils import Convolve, Common
    main()
