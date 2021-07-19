from math import sqrt
from numbers import Real
from typing import Union, List, Tuple

from numpy import std, mean
from pandas import Series
from scipy.stats import t, pearsonr


def get_repeated_cv_corrected_dof():
    k = 10
    c = 10
    h = 2 / (1 / c + 1 / k)
    return h


def confidence_interval(
        sample_values: List[Real],
        degrees_of_freedom: int = None,
) -> Tuple[float, Tuple[float, float], float]:
    degrees_of_freedom = degrees_of_freedom if degrees_of_freedom is not None else len(
        sample_values
    ) - 1
    sample_mean = mean(sample_values)
    sample_standard_deviation = std(sample_values, ddof=len(sample_values) - degrees_of_freedom)
    standard_error = sample_standard_deviation / sqrt(degrees_of_freedom)
    # noinspection PyBroadException
    try:
        interval = t.interval(0.95, degrees_of_freedom, loc=sample_mean, scale=standard_error)
    except:
        interval = (0, 0)
    return sample_mean, interval, sample_standard_deviation


def round_digits(number: Union[float, int], digits=3) -> str:
    number = float(number)
    integers = len(str(abs(int(number))))
    decimals = max(0, digits - integers)
    s = str(round(number, decimals))
    return s.rstrip('0').rstrip('.') if '.' in s else s


def compute_r(series1: Series, series2: Series) -> float:
    return pearsonr(series1, series2)[0]
