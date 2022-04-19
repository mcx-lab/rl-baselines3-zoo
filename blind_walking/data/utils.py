import pathlib
from re import L


def getDataPath():
    curr_dir = pathlib.Path(__file__).absolute().parent
    return str(curr_dir)
