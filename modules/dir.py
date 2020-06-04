import os


class Directory:
    @staticmethod
    def create(dir_path: str):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
