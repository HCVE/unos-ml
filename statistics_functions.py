from typing import Union

from math import sqrt

from numpy import std, mean
from scipy.stats import t


def get_dof():
    k = 10
    c = 10
    h = 2 / (1 / c + 1 / k)
    return h


def confidence_interval(h, values):
    s = std(values)
    m = mean(values)
    std_error = s / sqrt(h)
    interval = t.interval(0.95, 10, loc=m, scale=std_error)
    return m, interval, s


def round_digits(number: Union[float, int], digits=3) -> str:
    number = float(number)
    integers = len(str(abs(int(number))))
    decimals = max(0, digits - integers)
    s = str(round(number, decimals))
    return s.rstrip('0').rstrip('.') if '.' in s else s
