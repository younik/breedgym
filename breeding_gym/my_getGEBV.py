import numpy as np


def my_getGEBV(*args):
    return [np.average(v) for v in args]
