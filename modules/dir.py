import os


class Directory:
    @staticmethod
    def create(dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
