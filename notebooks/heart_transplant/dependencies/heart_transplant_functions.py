from enum import Enum
from functools import partial
from numbers import Number
from typing import Tuple, Dict, List, Union, Iterable

import logging
import numpy
import numpy as np
from pandas import DataFrame, Series

from utils import balanced_range, get_ilocs_by_callback

SURVIVAL_DAYS_DEFAULT: int = 365

FEATURES_FOR_LOG_TRANSFORM = (
    'tbili', 'tbili_don', 'age_don', 'age', 'distance', 'newpra', 'ischtime',
    'hgt_cm_calc', 'hgt_cm_don_calc', 'wgt_kg_calc', 'wgt_kg_don_calc',
    'most_rcnt_creat', 'creat_trr', 'creat_don', 'bun_don',
    'sgot_don', 'sgpt_don', 'cmassratio'
)


def reverse_log_transform_row(row: Series) -> Series:
    new_row = row.copy()
    for base_feature_name in FEATURES_FOR_LOG_TRANSFORM:
        target_feature_name = 'log_' + base_feature_name
        try:
            new_row[base_feature_name] = np.exp(row[target_feature_name]) - 1
            new_row.drop(target_feature_name, inplace=True)
        except FloatingPointError:
            print(target_feature_name, row[target_feature_name])
        except KeyError:
            logging.warning(f'Feature "{target_feature_name}" is not present')
    return new_row


def log_transform_dataset(dataset: DataFrame) -> DataFrame:
    new_dataset = dataset.copy()
    for log_plus_1_feature in FEATURES_FOR_LOG_TRANSFORM:
        new_dataset[
            f'log_{log_plus_1_feature}'
        ] = log_transform(dataset[log_plus_1_feature])
        new_dataset.drop(log_plus_1_feature, axis=1, inplace=True)

    return new_dataset


def log_transform(number: Number) -> float:
    # noinspection PyTypeChecker
    return np.log(number + 1)


def get_features(metadata: Dict) -> List[str]:
    return [key for key in metadata.keys() if not metadata[key].get('remove', False)]


def filter_out_unused_features(input_dataset: DataFrame, metadata: Dict) -> DataFrame:
    return input_dataset[get_features(metadata)]


def get_X_y_1_year_survival(
        input_dataset: DataFrame, survival_days: int = SURVIVAL_DAYS_DEFAULT
) -> Tuple[DataFrame, Series]:
    y = extract_y(input_dataset, survival_days=survival_days)
    defined_mask = ~y.isna()

    features_to_drop = []

    for feature_to_drop in ('death', 'futd', 'deathr', 'ptime', 'gtime', 'px_stat', 'pstatus', 'gstatus'):
        if feature_to_drop not in input_dataset.columns:
            logging.warning(f'Feature "{features_to_drop}" not present in the dataset')
        else:
            features_to_drop.append(feature_to_drop)

    return (
        input_dataset[defined_mask].drop(features_to_drop, axis=1),
        y[defined_mask].astype(int)
    )


def format_feature(metadata: Dict, feature_name: str) -> str:
    is_log = feature_name.startswith("log_")
    base_name = feature_name[len('log_'):] if is_log else feature_name
    try:
        return metadata[base_name]['name_long'] + (' (log)' if is_log else '')
    except KeyError:
        return feature_name


def format_features(metadata: Dict, features: Iterable[str]) -> List[str]:
    return [
        format_feature(metadata, feature) for feature in features
    ]


def format_feature_with_code(metadata: Dict, feature_name: str) -> str:
    is_log = feature_name.startswith("log_")
    base_name = feature_name[len('log_'):] if is_log else feature_name
    try:
        return metadata[base_name]['name_long'] + f' [{base_name}]' + (' (log)' if is_log else '')
    except KeyError:
        return feature_name


def format_columns(metadata: Dict, data_frame: DataFrame) -> DataFrame:
    data_frame_new = data_frame.copy()
    data_frame_new.columns = format_features(metadata, data_frame.columns)
    return data_frame_new


def get_y_from_preprocessed_row(row: Series, survival_days: int = SURVIVAL_DAYS_DEFAULT) -> Union[int, float]:
    numpy.seterr(all='raise')
    row_raw = reverse_log_transform_row(row)
    if row_raw['futd'] > survival_days:
        return 0
    elif row_raw['futd'] <= survival_days and (row_raw['death'] == 1):
        return 1
    else:
        return np.nan


def extract_y(input_dataset, survival_days: int = SURVIVAL_DAYS_DEFAULT):
    # noinspection PyTypeChecker
    y = input_dataset.apply(partial(get_y_from_preprocessed_row, survival_days=survival_days), axis=1)
    return y


def provide_long_labels(metadata: Dict, input_data: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
    input_data_copy = input_data.copy()
    input_data_copy.index = input_data_copy.index.map(lambda index: f'{metadata[index]["name_long"]} [{index}]')
    return input_data_copy


def get_expanding_windows(
        data_frame: DataFrame,
        test_size_years: int,
        minimum_training_years: int,
        n_windows: int = None,
        year_start: int = None,
        year_stop: int = None,
        validation_size_years: int = None
) -> Iterable[Tuple[List[int], ...]]:
    validation_size_years_int = 0 if validation_size_years is None else validation_size_years

    year_start = year_start if year_start is not None else (
            data_frame['tx_year'].min() + minimum_training_years + validation_size_years_int + test_size_years - 1
    )

    year_stop = year_stop if year_stop is not None else data_frame['tx_year'].max()

    if n_windows is None:
        n_windows = (year_stop - year_start) + 1

    thresholds = list(balanced_range(int(year_start), int(year_stop) + 1, int(n_windows)))

    for threshold in thresholds:
        yield (
            list(get_ilocs_by_callback(
                lambda row: row['tx_year'] <= threshold - test_size_years - validation_size_years_int, data_frame
            )),

            *([list(get_ilocs_by_callback(
                lambda row: (
                        threshold - test_size_years - validation_size_years_int
                        < row['tx_year']
                        <= threshold - test_size_years
                ), data_frame))] if validation_size_years is not None else []),

            list(get_ilocs_by_callback(
                lambda row: threshold - test_size_years < row['tx_year'] <= threshold,
                data_frame
            ))
        )


def remove_missing_columns(X: DataFrame, sampling_sets: Iterable, verbose: int = 0) -> DataFrame:
    removed_columns = set()
    for train_set, _ in sampling_sets:
        X_window = X.iloc[train_set]
        for column in X_window.columns:
            if len(X_window[column].value_counts()) <= 1:
                removed_columns.add(column)

    if verbose > 0:
        print('Removed features:', removed_columns)

    return X.drop(columns=list(removed_columns))


class AgeGroup(Enum):
    ALL = 'all'
    L_18 = 'l_18',
    ME_18 = 'me_18'


def get_filtered_by_age(group: AgeGroup, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    if group == 'all':
        X_filtered = X
    elif group == 'l_18':
        X_filtered = X[X['log_age'] < log_transform(18)]
    elif group == 'me_18':
        X_filtered = X[X['log_age'] >= log_transform(18)]
    else:
        raise Exception('Unknown group')

    y_filtered = y.loc[X_filtered.index]

    return X_filtered, y_filtered
