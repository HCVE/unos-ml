import logging
from functools import partial
from typing import Callable, Tuple, List

import numpy as np
import pandas
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from toolz import identity

from cache import memory
from functional import pipe, tap
from dependencies.heart_transplant_functions import log_transform_dataset, \
    filter_out_unused_features, get_X_y_1_year_survival, SURVIVAL_DAYS_DEFAULT, get_filtered_by_age, \
    remove_missing_columns, get_rolling_cv, AgeGroup, get_rolling_cv_nested
from dependencies.heart_transplant_metadata import heart_transplant_metadata as metadata
from dependencies.utils import get_nested_cv_sampling
from visualisation import format_number


def type_conversion(dataset):
    dataset_new = dataset.copy()
    dataset_new['height ratio'] = pandas.to_numeric(dataset['height ratio'])
    dataset_new['weight ratio'] = pandas.to_numeric(dataset['weight ratio'])
    dataset_new['tx_year'] = pandas.to_numeric(dataset['tx_year'])
    return dataset_new


def convert_missing_codes_to_na(_X):
    _X_new = _X.copy()
    for column in _X.columns:
        try:
            metadata_record_na = metadata[column]['na_values']
        except KeyError:
            pass
        else:
            _X_new[column].replace(metadata_record_na, np.nan, inplace=True)
    return _X_new


def transplant_occurred(_X):
    return _X[~_X['tx_year'].isna()]


def years_subset(_X, year_from, year_to):
    return _X[(_X['tx_year'] >= year_from) & (_X['tx_year'] <= year_to)]


def keep_only_heart(_X: DataFrame) -> DataFrame:
    return _X[_X['organ'] == 'HR']


def remove_with_missing_variables(_X):
    missing_mask = _X.copy().swifter.apply(lambda x: x.count() >= 80, axis=1)
    return _X[missing_mask]


def get_binary_dataset(survival_days: int = SURVIVAL_DAYS_DEFAULT, log_transform: bool = True):
    dataset_1, dataset_raw = get_base_dataset()
    dataset_2 = pipe(
        dataset_1,
        tap(lambda d: logging.info(f'Input features {format_number(len(d.columns))}')),
        log_transform_dataset if log_transform else identity,
    )

    X, y = get_X_y_1_year_survival(dataset_2, dataset_raw, survival_days=survival_days)
    return X, y, dataset_raw


def get_base_dataset():
    dataset_raw = load_dataset()
    return (
        pipe(
            dataset_raw,
            tap(lambda d: logging.info(f'Raw dataset, n= {format_number(len(d))}')),
            convert_missing_codes_to_na,
            type_conversion,
            partial(years_subset, year_from=1994, year_to=2016),
            tap(lambda d: logging.info(f'Years subset, n= {format_number(len(d))}')),
            keep_only_heart,
            tap(lambda d: logging.info(f'Only heart, n= {format_number(len(d))}')),
            partial(filter_out_unused_features, metadata=metadata),
        ),
        dataset_raw,
    )


def load_dataset():
    dataset_raw = pandas.read_csv("../unos-data/unos.csv")
    dataset_raw.columns = [column.lower() for column in dataset_raw.columns]
    return dataset_raw


def get_survival_dataset(survival_days: int = 365):
    X, y, dataset_raw = get_reduced_binary_dataset(survival_days=survival_days, log_transform=True)
    return X, y, dataset_raw['futd'].loc[X.index], dataset_raw['death'].loc[X.index], dataset_raw


get_survival_dataset_cached = memory.cache(get_survival_dataset)


def get_reduced_binary_dataset(
    survival_days: int = SURVIVAL_DAYS_DEFAULT, log_transform: bool = True
):
    X, y, dataset_raw = get_binary_dataset(survival_days=survival_days, log_transform=log_transform)

    X_missing_rows_removed = remove_with_missing_variables(X)

    print(f'Row with < 80 variables removed, n= {format_number(len(X_missing_rows_removed))}')

    return X_missing_rows_removed, y.loc[X_missing_rows_removed.index], dataset_raw


def reduce_dataset(X, y):
    missing_mask = X.copy().swifter.apply(lambda x: x.count() >= 80, axis=1)
    X_reduced = X[missing_mask]
    y_reduced = y[missing_mask]
    return X_reduced, y_reduced


get_reduced_binary_dataset_cached = memory.cache(get_reduced_binary_dataset)
get_binary_dataset_cached = memory.cache(get_binary_dataset)


def get_base_inputs(
    get_sampling_sets,
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
    filter_callback: Callable = identity,
) -> Tuple[DataFrame, Series, DataFrame, DataFrame, List[List]]:
    X, y, dataset_raw = get_reduced_binary_dataset_cached(survival_days)

    logging.info(f'Loaded n={len(X)}')

    X_filtered, y_filtered = get_filtered_by_age(group, X, y)
    X_filtered = filter_callback(X_filtered)
    y_filtered = y.loc[X_filtered.index]

    logging.info(f'Filtered, n={len(X_filtered)}')

    sampling_sets = get_sampling_sets(X_filtered, y_filtered, dataset_raw)

    logging.info(f'Cross-validation folds n={sampling_sets}')

    X_valid = remove_missing_columns(X_filtered, sampling_sets, verbose=1)

    logging.info(
        f'Removing features with 0 variance in some fold: {len(X_filtered.columns)} ➡ {len(X_valid.columns)}'
    )

    return X_filtered, y_filtered, X_valid, dataset_raw, sampling_sets


def get_rolling_cv_inputs(
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
) -> Tuple[DataFrame, Series, DataFrame, DataFrame, List[List]]:
    return get_base_inputs(
        lambda X, y, dataset_raw: list(
            get_rolling_cv(
                X.assign(tx_year=dataset_raw['tx_year']),
                n_windows=None,
                test_size_years=1,
                minimum_training_years=10,
                year_stop=2016
            )
        ),
        survival_days=survival_days,
        group=group,
    )


get_rolling_cv_inputs_cached = memory.cache(get_rolling_cv_inputs)


def get_nested_rolling_cv_inputs(
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
):
    return get_base_inputs(
        lambda X, y, dataset_raw: list(
            get_rolling_cv_nested(
                X.assign(tx_year=dataset_raw['tx_year']),
                n_windows=None,
                test_size_years=1,
                minimum_training_years=10,
                year_stop=2016,
                validation_size_years=1,
            )
        ),
        survival_days=survival_days,
        group=group,
    )


get_nested_rolling_cv_inputs_cached = memory.cache(get_nested_rolling_cv_inputs)

def get_shuffled_cv_inputs(
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
):
    return get_base_inputs(
        lambda X, y, dataset_raw: list(KFold(n_splits=10, shuffle=True).split(X, y)),
        survival_days=survival_days,
        group=group,
    )


get_shuffled_cv_inputs_cached = memory.cache(get_shuffled_cv_inputs)


def get_nested_shuffled_cv_inputs(
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
    group: AgeGroup = AgeGroup.ALL,
):
    return get_base_inputs(
        lambda X, y, dataset_raw: list(get_nested_cv_sampling(X)),
        survival_days=survival_days,
        group=group
    )


get_nested_shuffled_cv_inputs_cached = memory.cache(get_nested_shuffled_cv_inputs)
