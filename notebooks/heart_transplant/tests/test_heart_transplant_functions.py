from collections import Iterable, Callable

from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
import numpy as np

from notebooks.heart_transplant.dependencies.heart_transplant_functions import get_X_y_1_year_survival, log_transform_dataset, \
    reverse_log_transform_row, get_expanding_windows, remove_missing_columns, FEATURES_FOR_LOG_TRANSFORM


def test_get_X_y():
    X, y = get_X_y_1_year_survival(DataFrame({'death': [0, 0, 1, 1], 'futd': [5, 600, 5, 600], 'a': [1, 2, 3, 4]}))

    assert_frame_equal(
        X,
        DataFrame({
            'a': [2, 3, 4],
        }, index=[1, 2, 3]),
        check_like=True,
    )

    assert_series_equal(y, Series([0, 1, 0], index=[1, 2, 3]))


def test_log_transform():
    input_data = DataFrame(
        {key: [np.exp(2) - 1, np.exp(3) - 1, np.exp(4) - 1, np.exp(5) - 1] for key in FEATURES_FOR_LOG_TRANSFORM}
    )
    output_data = log_transform_dataset(input_data)

    assert_series_equal(
        output_data['log_tbili'],
        Series([2., 3., 4., 5.], name='log_tbili')
    )

    assert_series_equal(
        input_data.iloc[0],
        reverse_log_transform_row(log_transform_dataset(input_data).iloc[0])
    )


def test_get_expanding_windows():
    assert list(get_expanding_windows(
        DataFrame({
            'tx_year': list(range(11))
        }), n_windows=3, test_size_years=2, minimum_training_years=5
    )) == [
               ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10]),
               ([0, 1, 2, 3, 4, 5, 6], [7, 8]),
               ([0, 1, 2, 3, 4], [5, 6]),
           ]
    assert \
        list(get_expanding_windows(DataFrame({
            'tx_year': [10, 10, 10, 11, 11, 11, 12, 12]
        }), n_windows=1, test_size_years=2, minimum_training_years=1)) == [([0, 1, 2], [3, 4, 5, 6, 7])]

    assert \
        [
            ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10], [11, 12]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8], [9, 10]),
            ([0, 1, 2, 3, 4], [5, 6], [7, 8])
        ] == list(get_expanding_windows(DataFrame({
            'tx_year': list(range(13))
        }), n_windows=3, test_size_years=2, minimum_training_years=5, validation_size_years=2))

    assert \
        list(get_expanding_windows(DataFrame({
            'tx_year': [10, 10, 11, 12, 12, 13, 13, 13]
        }), n_windows=1, test_size_years=1, minimum_training_years=1)) == [
            ([0, 1, 2, 3, 4], [5, 6, 7])
        ]

    assert \
        list(get_expanding_windows(DataFrame({
            'tx_year': [10, 10, 11, 12, 12, 13, 13, 13]
        }), n_windows=None, test_size_years=1, minimum_training_years=1)) == [
            ([0, 1, 2, 3, 4], [5, 6, 7]),
            ([0, 1, 2], [3, 4]),
            ([0, 1], [2]),
        ]


def test_remove_missing_columns():
    df = DataFrame({'a': [np.nan, 1, np.nan, 2], 'b': [1, 1, 1, 1], 'c': [1, 2, 3, 2]})
    assert_frame_equal(df.drop(columns=['b']), remove_missing_columns(df, sampling_sets=[([0, 1, 2, 3], [])]))
    assert_frame_equal(df.drop(columns=['a', 'b']), remove_missing_columns(df, sampling_sets=[([0, 1], [])]))
