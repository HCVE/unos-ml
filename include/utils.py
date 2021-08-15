import asyncio
import logging
import os
import random
import shelve
import time
import timeit
from datetime import datetime, timedelta
from enum import Enum
from functools import partial, singledispatch
from functools import singledispatch
from numbers import Number
from pathlib import Path
from typing import List, Dict, Any, Iterable, Callable, Optional, TypeVar, Union, Iterator, Tuple, Set, Mapping, \
    TypedDict
from typing import Mapping

import numpy
import numpy as np
import pandas
from filelock import FileLock
from frozendict import frozendict
from functional_pipeline import pipeline
from numpy import linspace
from pandas import DataFrame, Series
from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from toolz import valmap
from toolz.curried import keyfilter

from include.cache import memory
from include.contants import FEATURES_TO_DROP
from include.custom_types import NestedCV, NestedTrainSet, CVSampling
from include.functional import statements, pipe
# noinspection PyUnresolvedReferences
import swifter

T1 = TypeVar('T1')
T2 = TypeVar('T2')

COLOR_BLIND_PALETTE = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
    '#dede00'
]


class LockedShelve:

    def __init__(self, path: str, *args, **kwargs):
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.lock = None
        self.shelve = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> 'LockedShelve':
        try:
            self.shelve.close()
        except AttributeError:
            pass
        data_path = Path(self.path)
        lock_folder = data_path.parent / '.lock'
        lock_folder.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(lock_folder / data_path.name)
        self.lock.acquire()
        self.shelve = shelve.open(self.path, *self.args, **self.kwargs)
        return self

    def close(self):
        self.shelve.close()
        self.shelve = None
        self.lock.release()

    def __setitem__(self, key, item):
        self.shelve[key] = item

    def __getitem__(self, key):
        return self.shelve[key]

    def __repr__(self):
        return repr(self.shelve)

    def __len__(self):
        return len(self.shelve)

    def __delitem__(self, key):
        del self.shelve[key]

    def clear(self):
        return self.shelve.clear()

    def copy(self):
        return self.shelve.copy()

    def has_key(self, k):
        return k in self.shelve

    def update(self, *args, **kwargs):
        return self.shelve.update(*args, **kwargs)

    def keys(self):
        return self.shelve.keys()

    def values(self):
        return self.shelve.values()

    def items(self):
        return self.shelve.items()

    def __iter__(self):
        return iter(self.shelve)


class Breakpoint(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None, **fit_params):
        breakpoint()
        return X

    @staticmethod
    def transform(X):
        return X

    def fit(self, **_):
        return self


def indent(how_much: int, what: str) -> str:
    return '\t' * how_much + what


def explore(structure: Any, indent_number: int = 0) -> None:
    current_indent = partial(indent, indent_number)
    if isinstance(structure, Mapping):
        for key in structure.keys():
            print(current_indent(key))
            explore(structure[key], indent_number + 1)
    elif isinstance(structure, List):
        print(current_indent('List of'))
        try:
            explore(structure[0], indent_number + 1)
        except IndexError:
            pass
    else:
        print(current_indent(str(type(structure))))


def get_object_attributes(something: object) -> List[str]:
    return [key for key in something.__dict__.keys() if not key.startswith('_')]


@singledispatch
def object2dict(obj) -> Dict:
    if hasattr(obj, '__dict__'):
        return object2dict(obj.__dict__)
    else:
        return obj


def get_class_attributes(cls: Any) -> List[str]:
    return list((k for k in cls.__annotations__.keys() if not k.startswith('__')))


def random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


class RandomSeed:

    def __init__(self, seed: Any) -> None:
        self.replace_seed = seed

    def __enter__(self) -> None:
        if self.replace_seed is not None:
            self.previous_np_seed = np.random.get_state()
            self.previous_random_seed = random.getstate()
        random_seed(self.replace_seed)

    def __exit__(self, *args, **kwargs) -> None:
        pass


def evaluate_and_assign_if_not_present(
    data: Union[Mapping, LockedShelve],
    key: str,
    callback: Callable[[], T1],
    was_not_present_callback: Callable[[T1], None] = None,
    force_execute: bool = False
) -> None:
    if isinstance(data, LockedShelve):
        fetched_data = data.open()
    else:
        fetched_data = data
    if key not in fetched_data or force_execute:
        logging.info(key.upper())
        logging.debug(f'Key "{key}" not present, executing callback')
        timer = Timer()
        try:
            fetched_data.close()
        except AttributeError:
            pass
        output = callback()
        if isinstance(data, LockedShelve):
            fetched_data = data.open()
        else:
            fetched_data = data
        fetched_data[key] = output
        try:
            fetched_data.close()
        except AttributeError:
            pass
        if was_not_present_callback:
            was_not_present_callback(data)
        logging.info(timer)
    else:
        logging.debug(f'Skipping "{key}", key present')


global_log_level = 1


def get_log_level():
    global global_log_level
    return global_log_level


def log(*args, level=0):
    global global_log_level
    if level <= global_log_level:
        print(shorten('[%s] %s' % (datetime.now().strftime('%H:%M:%S'), ' '.join(map(str, args)))))


def qualified_name(o):
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__
    else:
        return module + '.' + o.__name__


def shorten(text: str) -> str:
    return text[:200] + '...' if len(text) > 200 else text


def use_df_fn(
    input_data_frame: DataFrame,
    output_data: Any,
    reuse_columns=True,
    reuse_index=True,
    columns: Optional[List] = None
) -> DataFrame:
    df_arguments = {}
    if reuse_columns:
        df_arguments['columns'] = columns if columns is not None else input_data_frame.columns
    if reuse_index:
        df_arguments['index'] = input_data_frame.index
    if isinstance(output_data, csr_matrix):
        output_data = output_data.toarray()
    return DataFrame(output_data, **df_arguments)


class DFWrapped:

    def get_feature_names(self):
        try:
            super().get_feature_names()
        except AttributeError:
            return self.fitted_feature_names

    def get_fitted_feature_names(self, X: DataFrame, X_out: DataFrame = None):
        try:
            return X_out.columns
        except AttributeError:
            try:
                return super().get_feature_names()
            except (AttributeError, ValueError) as e:
                if isinstance(self, ColumnTransformer):
                    raise e
                try:
                    return X.columns
                except AttributeError:
                    raise Exception(
                        'Cannot produce DataFrame with named columns: columns are not defined'
                    )

    def transform(self, X, *args, **kwargs):
        try:
            X_out = super().transform(X, *args, **kwargs)
            self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
            out = use_df_fn(X, X_out, columns=self.fitted_feature_names)
            return out
        except AttributeError:
            return X

    def fit_transform(self, X, *args, **kwargs):
        X_out = super().fit_transform(X, *args, **kwargs)
        self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
        out = use_df_fn(X, X_out, columns=self.fitted_feature_names)
        return out

    def fit_predict(self, X, y, *args, **kwargs):
        X_out = super().fit_predict(X, y, *args, **kwargs)
        self.fitted_feature_names = self.get_fitted_feature_names(X, X_out)
        return use_df_fn(X, X_out, columns=self.fitted_feature_names)

    def predict(self, X, *args, **kwargs) -> Series:
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        y_pred = super().predict(X, *args, **kwargs)
        return y_pred

    def predict_proba(self, X, *args, **kwargs) -> None:
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        y_proba = super().predict_proba(X, *args, **kwargs)
        return y_proba

    def fit(self, X, y, *args, **kwargs):
        self.fitted_feature_names = self.get_fitted_feature_names(X)
        super().fit(X, y, *args, **kwargs)
        return self


class Print(BaseEstimator, TransformerMixin):

    def __init__(self, what):
        super()
        self.what = what

    def transform(self, X, *args, **kwargs):
        print(self.what)
        return X

    def fit(self, *args, **kwargs):
        return self


class Passthrough(BaseEstimator, TransformerMixin):

    def fit(self, X, y, *args, **kwars):
        return self

    def transform(self, X, *args, **kwargs):
        return X


class DFPassthrough(DFWrapped, Passthrough):
    ...


class Debug(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        fit_callback: Callable[[DataFrame], Any] = None,
        transform_callback: Callable[[DataFrame], Any] = None,
        set_breakpoint_fit: bool = False
    ):
        self.set_breakpoint_fit = set_breakpoint_fit
        self.fit_callback = fit_callback
        self.transform_callback = transform_callback
        super()

    def transform(self, X):
        if self.transform_callback:
            self.transform_callback(X)
        else:
            print('transform', X)
        return X

    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X)
        else:
            print('fit', X)
        if self.set_breakpoint_fit:
            breakpoint()
        return self


def balanced_range(start: int, end: int, count: int = None) -> Iterable:
    count = count if count is not None else end - start
    return (round(n) for n in linspace(end - 1, start, count))


def get_ilocs_by_callback(row_callback: Callable[[Series], bool],
                          data_frame: DataFrame) -> Iterator[int]:
    return (nr for (nr, (_, row)) in enumerate(data_frame.iterrows()) if row_callback(row))


def map_columns(callback: Callable, data_frame: DataFrame) -> DataFrame:
    return data_frame.rename(callback, axis=1)


class Timer:
    """Measure time used."""

    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time_cpu = timeit.default_timer()
        self._start_time_real = time.time()

    def elapsed_cpu(self) -> float:
        return timeit.default_timer() - self._start_time_cpu

    def elapsed_real(self) -> float:
        return time.time() - self._start_time_real

    def print_elapsed_cpu(self, message=None):
        if message:
            print(message + ':', str(self))
        else:
            print(str(self))

    def __str__(self) -> str:
        cpu_time = self.elapsed_cpu()
        time_output = str(timedelta(seconds=round(cpu_time, self._round_ndigits)))
        return f'Time elapsed: {time_output}'


class Counter:
    count: int
    initial: int

    def __init__(self, count: int = 0):
        self.initial = count
        self.count = count

    def increase(self, number: int = 1):
        self.count += number


def nested_iterable_to_list(what):
    if hasattr(what, '__iter__') and (not hasattr(what, '__len__')):
        return [nested_iterable_to_list(element) for element in what]
    else:
        return what


T = TypeVar('T', DataFrame, Series)


@singledispatch
def data_subset_iloc(data_frame: T, subset_index: List[int]) -> T:
    return data_frame.iloc[subset_index]


def remove_prefix(prefix: str, input_str: str) -> str:
    if input_str.startswith(prefix):
        return input_str[len(prefix):]
    else:
        return input_str[:]


def get_nested_cv_sampling(
    X: DataFrame,
    split_outer_cv=KFold(n_splits=10, shuffle=True).split,
    split_inner_cv=KFold(n_splits=10, shuffle=True).split
) -> NestedCV:
    return pipe(
        split_outer_cv(X),
        partial(
            map, lambda split: [
                NestedTrainSet(
                    cv=statements(
                        (splits := list(split_inner_cv(list(split[0])))), [
                            (np.take(np.array(split[0]), train), np.take(np.array(split[0]), test))
                            for (train, test) in splits
                        ]
                    ),
                    all=split[0]
                ), split[1]
            ]
        ), list
    )


def list_recursive(input: Any) -> List:
    if isinstance(input, Mapping):
        return pipe(input, partial(valmap, list_recursive))
    elif isinstance(input, Iterable) and (not isinstance(input, str)):
        return pipe(input, partial(map, list_recursive), list)
    else:
        return input


empty_dict: Mapping = frozendict()
MappingValue = TypeVar('MappingValue')


def set_logging(level: int) -> None:
    logging.basicConfig(
        level=level, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S', force=True
    )


def remove_suffix(suffix: str, input_str: str) -> str:
    # suffix='' should not call self[:-0].
    if suffix and input_str.endswith(suffix):
        return input_str[:-len(suffix)]
    else:
        return input_str[:]


def capitalize_first(input_str: str) -> str:
    return input_str[0].upper() + input_str[1:]


SURVIVAL_DAYS_DEFAULT: int = 365
FEATURES_FOR_LOG_TRANSFORM = (
    'tbili', 'tbili_don', 'age_don', 'age', 'distance', 'newpra', 'ischtime', 'hgt_cm_calc',
    'hgt_cm_don_calc', 'wgt_kg_calc', 'wgt_kg_don_calc', 'most_rcnt_creat', 'creat_trr',
    'creat_don', 'bun_don', 'sgot_don', 'sgpt_don', 'cmassratio'
)


def reverse_log_transform_row(row: Series) -> Series:
    new_row = row.copy()
    for base_feature_name in FEATURES_FOR_LOG_TRANSFORM:
        target_feature_name = 'log_' + base_feature_name
        try:
            new_row[base_feature_name] = reverse_log_series(row[target_feature_name])
            new_row.drop(target_feature_name, inplace=True)
        except FloatingPointError:
            print(target_feature_name, row[target_feature_name])
        except KeyError:
            pass
    return new_row


def reverse_log_series(series: Series) -> Series:
    return np.exp(series) - 1


def reverse_log_transform_dataset(dataset: DataFrame) -> DataFrame:
    return dataset.swifter.apply(reverse_log_transform_row, axis=1)


def log_transform_dataset(dataset: DataFrame) -> DataFrame:
    new_dataset = dataset.copy()
    for log_plus_1_feature in FEATURES_FOR_LOG_TRANSFORM:
        new_dataset[f'log_{log_plus_1_feature}'] = log_transform(dataset[log_plus_1_feature])
        new_dataset.drop(log_plus_1_feature, axis=1, inplace=True)
    return new_dataset


def log_transform(number: Number) -> float:
    return np.log(number + 1)


def get_features(metadata: Dict) -> List[str]:
    return [key for key in metadata.keys() if not metadata[key].get('remove', False)]


def filter_out_unused_features(input_dataset: DataFrame, metadata: Dict) -> DataFrame:
    return input_dataset[get_features(metadata)].drop(columns=FEATURES_TO_DROP)


def get_X_y_1_year_survival(
    X: DataFrame,
    dataset_raw: DataFrame,
    survival_days: int = SURVIVAL_DAYS_DEFAULT
) -> Tuple[DataFrame, Series]:
    X_raw = dataset_raw.loc[X.index]
    input_dataset_transplant_occurred = X[~X_raw['death'].isna()]
    logging.info(
        f'Removing rows with missing outcome: {len(X)} -> {len(input_dataset_transplant_occurred)}'
    )
    y = extract_y_from_preprocessed(X_raw, survival_days=survival_days)
    defined_mask = ~y.isna()
    logging.info(
        f'Outcome not defined: {len(input_dataset_transplant_occurred)} -> {defined_mask.sum()}'
    )
    return X[defined_mask], y[defined_mask].astype(int)


def extract_y_from_preprocessed(input_dataset, survival_days: int = SURVIVAL_DAYS_DEFAULT):
    y = input_dataset.swifter.apply(
        partial(get_y_from_preprocessed_row, survival_days=survival_days), axis=1
    )
    return y


def get_y_from_preprocessed_row(row: Series,
                                survival_days: int = SURVIVAL_DAYS_DEFAULT) -> Union[int, float]:
    numpy.seterr(all='raise')
    row_raw = reverse_log_transform_row(row)
    return get_y_from_raw_row(row_raw, survival_days)


def get_y_from_raw_row(row_raw, survival_days):
    if row_raw['futd'] > survival_days:
        return 0
    elif row_raw['futd'] <= survival_days and row_raw['death'] == 1:
        return 1
    else:
        return np.nan


def get_rolling_cv(
    data_frame: DataFrame,
    test_size_years: int,
    minimum_training_years: int,
    n_windows: int = None,
    year_start: int = None,
    year_stop: int = None,
    validation_size_years: int = None
) -> CVSampling:
    validation_size_years_int = 0 if validation_size_years is None else validation_size_years
    year_start = year_start if year_start is not None else data_frame['tx_year'].min(
    ) + minimum_training_years + validation_size_years_int + test_size_years - 1
    year_stop = year_stop if year_stop is not None else data_frame['tx_year'].max()
    if n_windows is None:
        n_windows = year_stop - year_start + 1
    thresholds = list(balanced_range(int(year_start), int(year_stop) + 1, int(n_windows)))
    for threshold in thresholds:
        yield (
            list(
                get_ilocs_by_callback(
                    lambda row: row['tx_year'] <= threshold - test_size_years -
                    validation_size_years_int, data_frame
                )
            ), *(
                [
                    list(
                        get_ilocs_by_callback(
                            lambda row: threshold - test_size_years - validation_size_years_int <
                            row['tx_year'] <= threshold - test_size_years, data_frame
                        )
                    )
                ] if validation_size_years is not None else []
            ),
            list(
                get_ilocs_by_callback(
                    lambda row: threshold - test_size_years < row['tx_year'] <= threshold,
                    data_frame
                )
            )
        )


def get_rolling_cv_nested(
    data_frame: DataFrame,
    test_size_years: int,
    minimum_training_years: int,
    n_windows: int = None,
    year_start: int = None,
    year_stop: int = None,
    validation_size_years: int = None
) -> NestedCV:
    validation_size_years_int = 0 if validation_size_years is None else validation_size_years
    year_start = year_start if year_start is not None else data_frame['tx_year'].min(
    ) + minimum_training_years + validation_size_years_int + test_size_years - 1
    year_stop = year_stop if year_stop is not None else data_frame['tx_year'].max()
    if n_windows is None:
        n_windows = year_stop - year_start + 1
    thresholds = list(balanced_range(int(year_start), int(year_stop) + 1, int(n_windows)))
    result = []
    for threshold in thresholds:
        result.append(
            (
                NestedTrainSet(
                    cv=[
                        (
                            list(
                                get_ilocs_by_callback(
                                    lambda row: row['tx_year'] <= threshold - test_size_years -
                                    validation_size_years_int, data_frame
                                )
                            ),
                            list(
                                get_ilocs_by_callback(
                                    lambda row: (
                                        row['tx_year'] > threshold - test_size_years -
                                        validation_size_years_int
                                    ) & (row['tx_year'] <= threshold - test_size_years), data_frame
                                )
                            )
                        )
                    ],
                    all=list(
                        get_ilocs_by_callback(
                            lambda row: row['tx_year'] <= threshold - test_size_years, data_frame
                        )
                    )
                ),
                list(
                    get_ilocs_by_callback(
                        lambda row: threshold - test_size_years < row['tx_year'] <= threshold,
                        data_frame
                    )
                )
            )
        )
    return result


get_rolling_cv_cached = memory.cache(get_rolling_cv)


def remove_missing_columns(X: DataFrame, sampling_sets: Iterable, verbose: int = 0) -> DataFrame:
    removed_columns: Set = set()
    for (train_set, test_set) in sampling_sets:
        try:
            X_window_train = X.iloc[train_set]
        except TypeError:
            X_window_train = X.iloc[train_set['all']]
        removed_columns.update(get_irrelevant_columns_df(X_window_train))
    if verbose > 0:
        print('Removed features:', removed_columns)
    return X.drop(columns=list(removed_columns))


def get_irrelevant_columns_df(X):
    for column in X.columns:
        if len(X[column].value_counts()) <= 1:
            yield column


class AgeGroup(Enum):
    ALL = 'ALL'
    L_18 = ('L_18', )
    ME_18 = 'ME_18'


def get_filtered_by_age(
    group: AgeGroup,
    X: DataFrame,
    y: Series = None,
) -> Union[Tuple[DataFrame, Series], DataFrame]:
    if group == AgeGroup.ALL:
        X_filtered = X
    elif group == AgeGroup.L_18:
        X_filtered = X[X['log_age'] < log_transform(18)]
    elif group == AgeGroup.ME_18:
        X_filtered = X[X['log_age'] >= log_transform(18)]
    else:
        raise Exception('Unknown group')
    if y is not None:
        y_filtered = y.loc[X_filtered.index]
        return X_filtered, y_filtered
    else:
        return X_filtered


def get_survival_y(dataset_raw: DataFrame) -> DataFrame:
    return pipe(
        DataFrame(
            {
                'death': dataset_raw['death'],
                'futd': dataset_raw['futd']
            }, index=dataset_raw.index
        ),
        partial(
            pandas.DataFrame.to_records,
            index=False,
            column_dtypes={
                'death': np.bool_,
                'futd': np.int32
            }
        )
    )


def get_feature_metadata(feature_name: str, metadata: Mapping) -> Mapping:
    feature_name_base = feature_name[4:] if feature_name.startswith('log_') else feature_name
    return metadata[feature_name_base]


class DecomposedOneHotFeatureName(TypedDict):
    name: str
    value: Optional[str]


def get_final_feature_sets(X) -> Tuple[List[str], List[str]]:
    (categorical_features, continuous_features) = get_categorical_and_continuous_features(X)
    return (
        [feature for feature in categorical_features if feature in X.columns],
        [feature for feature in continuous_features if feature in X.columns]
    )


def remove_column_prefix(X: DataFrame) -> DataFrame:
    return map_columns(
        lambda column_name: pipe(
            column_name, partial(remove_prefix, 'categorical__'),
            partial(remove_prefix, 'continuous__')
        ), X
    )


def get_categorical_and_continuous_features(X: DataFrame) -> Tuple[List, List]:
    defined_categorical_features = [
        'diag', 'func_stat_trr', 'contin_cocaine_don', 'func_stat_tcr', 'retransplant', 'iabp_tcr',
        'inotropes_tcr', 'ethcat', 'hcv_serostatus', 'lvad ever', 'gender', 'gender_don',
        'tah ever', 'med_cond_trr', 'anyecmo', 'anyvent', 'diab', 'serostatus', 'ecmo', 'ecmo_trr',
        'gstatus', 'pstatus', 'ethcat_don', 'blood_inf_don', 'other_inf_don', 'pulm_inf_don',
        'urine_inf_don', 'cod_cad_don', 'death_mech_don', 'multiorg', 'abo_mat', 'lv_eject_meth',
        'coronary_angio', 'vessels_50sten', 'biopsy_dgn', 'ecd_donor', 'education', 'education',
        'congenital', 'prior_card_surg_trr', 'prior_card_surg_tcr', 'prior_card_surg_type_tcr',
        'ventilator_tcr', 'rvad ever', 'cancer_site_don'
    ]
    categorical_features = [
        feature_name for (feature_name, series) in X.items()
        if series.dtype == 'object' or feature_name in defined_categorical_features
    ]
    continuous_features = [
        feature_name for feature_name in X.columns if feature_name not in categorical_features
    ]
    return categorical_features, continuous_features


def get_categorical_and_continuous_features_one_hot(X: DataFrame) -> Tuple[List, List]:
    features = ['diag', 'func_stat_trr']
    (categorical_features, continuous_features) = get_categorical_and_continuous_features(X)
    for feature in features:
        try:
            categorical_features.remove(feature)
            continuous_features.append(feature)
        except ValueError:
            pass
    return categorical_features, continuous_features


class SurvivalWrapper:
    days = None

    def set_survival_days(self, days):
        self.days = days
        return self

    def predict_proba(self, X) -> DataFrame:
        try:
            y_score_0 = [
                fn(self.days) for fn in self.predict_survival_function(X, return_array=False)
            ]
        except TypeError:
            y_score_0 = [fn(self.days) for fn in self.predict_survival_function(X)]
        y_score_1 = [1 - y_score for y_score in y_score_0]
        return DataFrame(
            {
                'y_predict_probabilities_0': y_score_0,
                'y_predict_probabilities_1': y_score_1
            },
            index=X.index
        )


def decompose_one_hot_feature_name(
    input_identifier: str, metadata: Mapping
) -> DecomposedOneHotFeatureName:
    components = input_identifier.split('_')
    identifier_base = '_'.join(components[:-1])

    try:
        get_feature_metadata(input_identifier, metadata=metadata)
    except KeyError:
        pass
    else:
        return DecomposedOneHotFeatureName(name=input_identifier, value=None)

    for key, item in metadata.items():
        if key == identifier_base:
            return DecomposedOneHotFeatureName(name=identifier_base, value=components[-1])
    return DecomposedOneHotFeatureName(name=input_identifier, value=None)


def json_serialize_replace(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [json_serialize_replace(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: json_serialize_replace(v) for (k, v) in obj.items()}
    elif isinstance(obj, range):
        return list(obj)
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()
    elif isinstance(obj, numpy.integer):
        return int(obj)
    elif isinstance(obj, numpy.floating) and (not numpy.isnan(obj)):
        return float(obj)
    else:
        try:
            if numpy.isnan(float(obj)):
                return None
            else:
                return obj
        except (ValueError, TypeError):
            return obj


@singledispatch
def json_serialize_types(value):
    try:
        value.__dict__
    except AttributeError:
        return value
    return {
        '__type': type(value).__module__ + '.' + type(value).__name__,
        **pipeline(
            value.__dict__, [keyfilter(lambda l: not l.startswith('_')), json_serialize_types]
        )
    }


@singledispatch
def json_deserialize_types(value):
    return value


async def merge(*iterables):
    iter_next = {it.__aiter__(): None for it in iterables}
    while iter_next:
        for (it, it_next) in iter_next.items():
            if it_next is None:
                fut = asyncio.ensure_future(it.__anext__())
                fut._orig_iter = it
                iter_next[it] = fut
        (done, _) = await asyncio.wait(iter_next.values(), return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            iter_next[fut._orig_iter] = None
            try:
                ret = fut.result()
            except StopAsyncIteration:
                del iter_next[fut._orig_iter]
                continue
            yield ret


def get_folder(path: str) -> str:
    return os.path.dirname(os.path.realpath(path))


def configuration_to_params(dictionary: Dict) -> Dict:
    return_value = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                return_value["%s__%s" % (key, key2)] = value2
        else:
            return_value[key] = value

    return return_value
