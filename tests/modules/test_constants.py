import unittest

from modules.constants import Constants


class TestConstants(unittest.TestCase):
    def test_hoge(self):
        c = Constants()
        print(c.MODEL_FILE_PATH)
