import os
import time
from contextlib import contextmanager


class PerformanceLogger:
    def __init__(self, workdir: str):
        self.__workdir = workdir
        self.__records = []

    @contextmanager
    def step(self, name: str):
        started_at = time.perf_counter()
        before_size = self.__get_directory_size(self.__workdir)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started_at
            after_size = self.__get_directory_size(self.__workdir)
            delta_size = after_size - before_size
            record = {
                "name": name,
                "elapsed": elapsed,
                "size": after_size,
                "delta": delta_size,
            }
            self.__records.append(record)
            print(
                "[Metrics] {name}: elapsed={elapsed}, workdir_size={size}, delta={delta}".format(
                    name=name,
                    elapsed=self.__format_seconds(elapsed),
                    size=self.__format_bytes(after_size),
                    delta=self.__format_bytes(delta_size),
                )
            )

    def print_summary(self):
        print("[Metrics] Summary")
        for record in self.__records:
            print(
                "  - {name}: elapsed={elapsed}, workdir_size={size}, delta={delta}".format(
                    name=record["name"],
                    elapsed=self.__format_seconds(record["elapsed"]),
                    size=self.__format_bytes(record["size"]),
                    delta=self.__format_bytes(record["delta"]),
                )
            )
        print("[Metrics] Directory size breakdown")
        for path, size in self.__get_size_breakdown():
            print("  - {path}: {size}".format(path=path, size=self.__format_bytes(size)))

    def __get_size_breakdown(self):
        if not os.path.exists(self.__workdir):
            return []

        breakdown = [(self.__workdir, self.__get_directory_size(self.__workdir))]
        for name in sorted(os.listdir(self.__workdir)):
            path = os.path.join(self.__workdir, name)
            if os.path.isdir(path):
                breakdown.append((path, self.__get_directory_size(path)))
            elif os.path.isfile(path):
                breakdown.append((path, os.path.getsize(path)))
        return breakdown

    @staticmethod
    def __get_directory_size(path: str) -> int:
        if not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            return os.path.getsize(path)

        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
        return total

    @staticmethod
    def __format_seconds(seconds: float) -> str:
        return "{0:.2f}s".format(seconds)

    @staticmethod
    def __format_bytes(size: int) -> str:
        sign = "-" if size < 0 else ""
        size = abs(float(size))
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return "{0}{1:.0f}{2}".format(sign, size, unit)
                return "{0}{1:.2f}{2}".format(sign, size, unit)
            size /= 1024
