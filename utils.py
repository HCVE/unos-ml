import hashlib
import importlib
import json
import logging
import math
import os
import random
import re
import time
import timeit
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import partial
from functools import singledispatch
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Callable, Optional, TypeVar, Union, Iterator
from typing import Mapping

import joblib
import numpy as np
import pandas
import psutil
import subprocess

from frozendict import frozendict
from miceforest import MultipleImputedKernel
from numpy import linspace
from pandas import DataFrame
from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from toolz import pluck, keyfilter
# noinspection PyUnresolvedReferences
import swifter
from arguments import get_params
from cache import memory
from functional import statements, raise_exception, find_index, pipe

T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Breakpoint(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None, **fit_params):
        breakpoint()
        return X

    @staticmethod
    def transform(X):
        return X

    def fit(self, **_):
        return self


def ll(some):
    if callable(some):

        def inner(*args, **kwargs):
            print("Input", args, kwargs)
            res = some(*args, **kwargs)
            print("Output", res)
            return res

        return inner
    else:
        print(some)
        return some


def llb(some):
    if callable(some):

        def inner(*args, **kwargs):
            print("Input", args, kwargs)
            breakpoint()
            res = some(*args, **kwargs)
            print("Output", res)
            return res

        return inner
    else:
        print(some)
        breakpoint()
        return some


def pyplot_set_text_color(color: str) -> None:
    from matplotlib import pyplot as plt
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = color
    plt.rcParams['xtick.color'] = color
    plt.rcParams['ytick.color'] = color


def get_tabulate_format():
    tablefmt = 'fancy_grid'

    return dict(tablefmt=tablefmt, floatfmt=".3f")


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
    return [key for key in something.__dict__.keys() if not key.startswith("_")]


def get_list_of_objects_keys(something: List[Dict]) -> List[str]:
    return list(something[0].keys()) if len(something) > 0 else []


@singledispatch
def object2dict(obj) -> Dict:
    if hasattr(obj, '__dict__'):
        return object2dict(obj.__dict__)
    else:
        return obj


@object2dict.register  # mypy: ignore
def _1(obj: list):
    return [object2dict(item) for item in obj]


@object2dict.register  # mypy: ignore
def _2(obj: Series):
    return obj.item()


@object2dict.register  # mypy: ignore
def _3(obj: dict):
    return {key: object2dict(item) for key, item in obj.items() if not key.startswith('__')}


def get_class_attributes(cls: Any) -> List[str]:
    # noinspection PyTypeChecker
    return list((k for k in cls.__annotations__.keys() if not k.startswith('__')))


def transpose_iter_of_dicts(list_of_dicts: Iterable[Dict]) -> Dict[Any, Iterable]:
    try:
        first_item = next(iter(list_of_dicts))
    except StopIteration:  # Length of 0
        return {}

    return {key: pluck(key, list_of_dicts) for key in first_item.keys()}


def transpose_dicts(input_dict: Dict[Any, Dict]) -> Dict[Any, Dict]:
    try:
        first_item = next(iter(input_dict.values()))
    except StopIteration:  # Length of 0
        return {}

    return {
        key: {key2: value2[key]
              for key2, value2 in input_dict.items()}
        for key in first_item.keys()
    }


def transpose_list(input_list: List[List]) -> List[List]:
    return list(map(list, zip(*input_list)))


def random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def from_items(pairs: Iterable[Tuple]) -> Dict:
    return {key: value for key, value in pairs}


def assert_equals(variable1: Any, variable2: Any) -> None:
    if variable1 == variable2:
        return
    else:
        raise AssertionError(
            f'Does not equal\nLeft side: {str(variable1)}\nRight side: {str(variable2)}'
        )


def assert_objects_equal(actual: Any, expected: Any) -> None:
    try:
        actual__dict__ = vars(actual)
        expected__dict__ = vars(expected)
    except TypeError:
        actual__dict__ = actual
        expected__dict__ = expected

    all_keys = set(actual__dict__.keys()).union(expected__dict__.keys())
    for key in all_keys:
        actual_value = actual__dict__[key]
        expected_value = expected__dict__[key]

        if isinstance(actual_value, DataFrame):
            assert_frame_equal(actual_value, expected_value)
        elif isinstance(actual_value, Series):
            assert_series_equal(actual_value, expected_value)
        else:
            assert_equals(actual_value, expected_value)


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


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=4, sort_keys=True)


def evaluate_and_assign_if_not_present(
    data: Union[Mapping, Callable],
    key: str,
    callback: Callable[[], T1],
    was_not_present_callback: Callable[[T1], None] = None,
    catch_exception: bool = False,
    force_execute: bool = False
) -> None:

    if isinstance(data, Callable):
        fetched_data = data()
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

        if isinstance(data, Callable):
            fetched_data = data()
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


def get_project_dir() -> Optional[str]:
    current_path = Path.cwd()
    for parent_folder in current_path.parents:
        if (parent_folder / 'Pipfile').is_file():
            return str(parent_folder.absolute())
    return None


def warning(*args):
    print(
        shorten(
            "WARNING [%s]: %s" % (datetime.now().strftime("%H:%M:%S"), " ".join(map(str, args)))
        )
    )


global_log_level = 1


def set_log_level(level: int) -> None:
    global global_log_level
    global_log_level = level


def get_log_level():
    global global_log_level
    return global_log_level


def log(*args, level=0):
    global global_log_level
    if level <= global_log_level:
        print(shorten("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), " ".join(map(str, args)))))


def train_test_model(model, test_size=0.8, features=None, label="LVDDF"):
    data = load_input_data()

    X, y = extract_features_and_label(data, label, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    model.train(X_train, y_train)

    return model.test(X_test, y_test)


# noinspection PyUnresolvedReferences
def get_model_class(model):
    module = importlib.import_module('methods.' + model)
    return module.get_class()


def merge_config_with_default(config=None):
    return {
        **load_global_config(),
        **(config if config else {}),
    }


def get_features_from_config(config):
    return config.get("features") or config["available_features"]


def get_label_from_config(config):
    return config.get("label").upper()


def load_input_data(path=None) -> DataFrame:
    data_frame = pandas.read_csv(path or "%s/data/data.csv" % get_params('data_folder'))
    data_frame.columns = [column.upper() for column in data_frame.columns]
    return data_frame


def load_dictionary():
    with open("%s/output/dictionary.json" % get_params('data_folder'), "r") as f:
        return json.load(f)


def load_global_config():
    with open("%s/default_parameters.json" % get_params('data_folder'), "r") as f:
        return json.load(f)


def load_data(file_name: str) -> Any:
    with open(f'{get_params("data_folder")}/{file_name}.json', "r") as f:
        return json.load(f)


def extract_features_and_label(
    data,
    label: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Tuple[DataFrame, Optional[Series]]:
    if label:
        label_capitalized = label.upper()
        if features:
            X = extract_features(data, features)
        else:
            X = data.drop([label_capitalized], axis=1)

        y = data[label_capitalized]
        return X, y
    else:
        if features:
            X = extract_features(data, features)
        else:
            X = data
        return X, None


def extract_features(data, features):
    adjusted_features = get_present_features(data, features)
    return data[list(adjusted_features)]


def get_present_features(data, features):
    from formatting import format_feature

    extracted = [feature.upper() for feature in features]
    for feature in features:
        feature_uppercase = feature.upper()
        if feature_uppercase not in data:
            warning("Attribute '%s' not in the dataframe" % format_feature(feature_uppercase))
            extracted.remove(feature_uppercase)
    return extracted


def get_feature_category_from_dictionary(feature_name, dictionary):
    return dictionary[feature_name]["category"]


def get_feature_category(feature_name):
    return get_feature_category_from_dictionary(
        feature_name,
        load_dictionary(),
    )


def qualified_name(o):
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__name__


def make_instance(cls):
    # noinspection PyBroadException
    try:
        return cls()
    except Exception:
        actual_init = cls.__init__
        cls.__init__ = lambda *args, **kwargs: None
        instance = cls()
        cls.__init__ = actual_init
        return instance


def await_func(func, *args, **kwargs):
    asyncio.run(func(*args, **kwargs))


def nice(nr: int = 10):
    process = psutil.Process(os.getpid())
    import sys
    try:
        getattr(sys, 'getwindowsversion')()
    except AttributeError:
        isWindows = False
    else:
        isWindows = True

    if isWindows:
        process.nice(getattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'))
    else:
        process.nice(nr)


def ignore_futures():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def shorten(text: str) -> str:
    return (text[:200] + '...') if len(text) > 200 else text


def get_row_number(index: int, columns: int) -> int:
    return math.ceil(index / columns)


def get_column_number(index: int, columns: int) -> int:
    return index % columns


def is_last_row(index: int, rows: int, columns: int) -> bool:
    return rows == get_row_number(index + 1, columns)


def is_first_column(index: int, columns: int) -> bool:
    return get_column_number(index, columns) == 0


# noinspection PyTypeChecker
def data_frame_count_na(data_frame: DataFrame) -> DataFrame:
    return data_frame.apply(lambda column: series_count_na(column))


def series_count_na(series: Series) -> int:
    distribution = series.isna().value_counts()
    try:
        return int(distribution[True])
    except KeyError:
        return 0


def use_df_fn(
    input_data_frame: DataFrame,
    output_data: Any,
    reuse_columns=True,
    reuse_index=True,
    columns: Optional[List] = None,
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
            return super().get_feature_names()
        except AttributeError:
            return self.columns

    def _get_columns(self, X: DataFrame, X_out: DataFrame = None):
        try:
            return self.get_feature_names()
        except AttributeError:
            try:
                return X_out.columns
            except AttributeError:
                try:
                    return X.columns
                except AttributeError:
                    raise Exception(
                        'Cannot produce DataFrame with named columns: columns are not defined'
                    )

    def transform(self, X, *args, **kwargs):
        try:
            X_out = super().transform(X, *args, **kwargs)
            self.columns = self._get_columns(X, X_out)
            return use_df_fn(X, X_out, columns=self.columns)
        except AttributeError:
            return X

    def fit_transform(self, X, y=None, *args, **kwargs):
        try:
            super().fit_transform
        except AttributeError:
            return super().fit(X, y, *args, **kwargs)
        else:
            X_out = super().fit_transform(X, y, *args, **kwargs)
            self.columns = self._get_columns(X, X_out)
            return use_df_fn(X, X_out, columns=self.columns)

    def fit_predict(self, X, y, *args, **kwargs):
        X_out = super().fit_predict(X, y, *args, **kwargs)
        self.columns = self._get_columns(X, X_out)
        return use_df_fn(X, X_out, columns=self.columns)

    def fit(self, X, y, *args, **kwargs):
        self.columns = self._get_columns(X)
        super().fit(X, y, *args, **kwargs)
        return self


def series_count_inf(series: Series) -> int:
    distribution = np.isinf(series)
    try:
        return int(distribution[True])
    except KeyError:
        return 0


def preliminary_analysis(dataset, show_missing=True, show_infinite=True, show_all=False):
    for column_name, series in dataset.items():
        n_missing = series_count_na(series)
        if series.dtype != 'object':
            n_inf = series_count_inf(series)
        else:
            n_inf = 0

        if (n_missing > 0 and show_missing) or (n_inf > 0 and show_infinite) or show_all:
            print(column_name)
            print(series.dtype)

            print(f'Missing: {n_missing}')

            if series.dtype != 'object':
                print(f'Inf: {n_inf}')
            print()


def generate_type_variants(value: Any) -> List[Any]:
    variants = {value}

    try:
        variants.add(float(value))
    except ValueError:
        pass

    try:
        variants.add(int(value))
    except ValueError:
        pass

    try:
        variants.add(str(value))
    except ValueError:
        pass

    return list(variants)


def with_default(value: T1, default: T2) -> Union[T1, T2]:
    return value if value is not None else default


def preliminary_analysis_column(input_data: Series) -> None:
    types = [type(index) for index in input_data.value_counts().index]
    descriptive_df = DataFrame({
        'types': types,
        'values': input_data.value_counts(),
    })
    print(descriptive_df)


def inverse_binary_y(y: Series) -> Series:
    # noinspection PyTypeChecker
    return 1 - y


class Print(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        what,
    ):
        super()
        self.what = what

    def transform(self, X, *args, **kwargs):
        print(self.what)
        return X

    def fit(self, *args, **kwargs):
        return self


class Debug(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        fit_callback: Callable[[DataFrame], Any] = None,
        transform_callback: Callable[[DataFrame], Any] = None,
        set_breakpoint_fit: bool = False,
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

    # noinspection PyUnusedLocal
    def fit(self, X, y=None, **fit_params):
        if self.fit_callback:
            self.fit_callback(X)
        else:
            print('fit', X)
        if self.set_breakpoint_fit:
            breakpoint()
        return self


def debug_item():
    return ('debug', Debug(fit_callback=lambda r: print(r))),


def balanced_range(start: int, end: int, count: int = None) -> Iterable:
    count = count if count is not None else (end - start)
    print(start, end, count)
    return (round(n) for n in linspace(end - 1, start, count))


def get_ilocs_by_callback(row_callback: Callable[[Series], bool],
                          data_frame: DataFrame) -> Iterator[int]:
    return (nr for nr, (_, row) in enumerate(data_frame.iterrows()) if row_callback(row))


def map_index(callback: Callable, data_frame: DataFrame) -> DataFrame:
    data_frame_new = data_frame.copy()
    data_frame_new.index = list(map(callback, list(data_frame.index)))
    return data_frame_new


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
            print(message + ":", str(self))
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
    if hasattr(what, '__iter__') and not hasattr(what, '__len__'):
        return [nested_iterable_to_list(element) for element in what]
    else:
        return what


def hash_dict(dictionary: Mapping) -> str:
    unique_str = str.encode(
        ''.join(["'%s':'%s';" % (key, val) for (key, val) in sorted(dictionary.items())])
    )
    return hashlib.sha1(unique_str).hexdigest()


def list_of_dicts_to_dict_of_lists(ld: List[Dict]) -> Dict[Any, List]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def dict_mean(iterable: Iterable[dict]) -> dict:
    mean_dict = {}
    iterable_list = list(iterable)
    for key in iterable_list[0].keys():
        mean_dict[key] = sum(d[key] for d in iterable_list) / len(iterable_list)
    return mean_dict


def call_and_push_to_queue(partial_func, queue):
    ret = partial_func()
    queue.put(ret)


def install_r_package(packages):
    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('biomarkers_utils')
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
    from rpy2.robjects.vectors import StrVector
    utils.install_packages(StrVector(packages))


def get_class_ratio(series: Series) -> float:
    class_counts = series.value_counts()
    return class_counts[0] / class_counts[1]


def compress_data_frame(input_dataframe: DataFrame) -> DataFrame:
    print(DataFrame(input_dataframe.to_numpy(), columns=list(input_dataframe.columns)).dtypes)
    print(DataFrame(input_dataframe.to_numpy()).dtypes)
    return DataFrame(input_dataframe.to_numpy(), columns=list(input_dataframe.columns))


def get_feature_subset(features: Iterable[str], data_frame: DataFrame) -> DataFrame:
    real_features = [
        feature if (feature in data_frame.columns) else (
            statements(
                log_feature := f'log_{feature}', log_feature if log_feature in data_frame.columns
                else raise_exception(KeyError(f'Feature {feature} not present'))
            )
        ) for feature in features
    ]
    return data_frame[real_features]


def append_to_pipeline(after_key: str, what: Tuple[str, Any], pipeline: Pipeline) -> Pipeline:
    return type(pipeline)(steps=append_to_tuples(after_key, what, pipeline.steps))


def append_to_tuples(after_key: str, what: Any, tuples: List[Tuple[str,
                                                                   Any]]) -> List[Tuple[str, Any]]:
    print(tuples)
    index = find_index(lambda pair: pair[0] == after_key, tuples)
    tuples_new = tuples.copy()
    tuples_new.insert(index + 1, what)
    return tuples_new


def encode_dict_to_params(input_dict: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    output_dict = {}

    for key1, value1 in input_dict.items():
        if isinstance(value1, Mapping):
            for key2, value2 in value1.items():
                output_dict[f'{key1}__{key2}'] = value2
        else:
            output_dict[key1] = value1

    return output_dict


def remove_newlines(value: str) -> str:
    return re.sub('\\\\n', '', value)


def execute(command: List[str]) -> Any:
    return subprocess.run(command, stdout=subprocess.PIPE).stdout


def compute_matrix_from_columns(
    input_df: DataFrame, callback: Callable, remove_na: bool = True
) -> DataFrame:
    output_data = OrderedDict.fromkeys(input_df.columns)

    for feature1_name, feature_1_series in input_df.items():
        output_data[feature1_name] = OrderedDict.fromkeys(input_df.columns)

        for feature2_name, feature_2_series in input_df.items():

            if remove_na:
                mask = ~feature_1_series.isna() & ~feature_2_series.isna()
            else:
                mask = [True] * len(feature_1_series)

            output_data[feature1_name][feature2_name] = callback(
                feature_1_series[mask], feature_2_series[mask]
            )

    return DataFrame(output_data, columns=input_df.columns, index=input_df.columns)


def get_dtype_outliers(series: Series) -> List[Tuple[str, str]]:
    types = series.swifter.progress_bar(False).apply(lambda value: type(value))
    return types.value_counts().iloc[1:]


def df_matrix_to_pairs(data_frame: DataFrame) -> DataFrame:
    output = {key: [] for key in ('x', 'y', 'value')}

    for column1 in data_frame.columns:
        for column2 in data_frame.columns:
            output['x'].append(column1)
            output['y'].append(column2)
            output['value'].append(data_frame.loc[column1][column2])

    return DataFrame(output)


from memoization import cached


def fit_mice_hash(X: DataFrame, iterations: int = 1):
    return joblib.hash(X)


def fit_mice_execute(X: DataFrame, iterations: int = 1) -> Any:
    kernel = MultipleImputedKernel(X, save_all_iterations=True, datasets=1)
    kernel.mice(iterations, verbose=True)
    return kernel


fit_mice_execute_cached = memory.cache(fit_mice_execute)


def fit_mice_memory_cached_(X: DataFrame, iterations: int = 1) -> Any:
    print('MICE cache miss')
    return fit_mice_execute(X, iterations)


fit_mice_execute_memory_cached = cached(fit_mice_memory_cached_, custom_key_maker=fit_mice_hash)


class MiceForest(BaseEstimator, TransformerMixin):

    def __init__(self, iterations: int = 1):
        self.kernel = None
        self.iterations = iterations

    def fit(self, X, y=None, **fit_params):
        self.kernel = fit_mice_execute_memory_cached(X)
        return self

    def transform(self, X, y=None):
        print('MICE transforming')
        data = self.kernel.impute_new_data(X).complete_data(self.iterations)
        return data


@singledispatch
def data_subset_iloc(data_frame: DataFrame, subset_index: List[int]) -> DataFrame:
    return data_frame.iloc[subset_index]


@data_subset_iloc.register(Series)
def data_subset_iloc_(data_frame: Series, subset_index: List[int]) -> Series:
    return data_frame.iloc[subset_index]


@data_subset_iloc.register
def data_subset_iloc_(data_frame: np.recarray, subset_index: List[int]) -> np.recarray:
    return data_frame[subset_index]


def get_feature_indexes_by_names(
    all_features: List[str],
    selected_features: List[str],
) -> List[int]:
    return [all_features.index(selected_feature) for selected_feature in selected_features]


def remove_prefix(prefix: str, input_str: str) -> str:
    if input_str.startswith(prefix):
        return input_str[len(prefix):]
    else:
        return input_str[:]


def remove_suffix(suffix: str, input_str: str) -> str:
    # suffix='' should not call self[:-0].
    if suffix and input_str.endswith(suffix):
        return input_str[:-len(suffix)]
    else:
        return input_str[:]


def remove_suffixes(suffixes: List[str], input_str: str) -> str:
    processed_str = input_str
    for suffix in suffixes:
        processed_str = remove_suffix(suffix, processed_str)
    return processed_str


def mapping_subset(keys: Iterable[str], input_mapping: Mapping) -> Mapping:
    return pipe(input_mapping, partial(keyfilter, lambda key: key in keys))


empty_dict: Mapping = frozendict()

if __name__ == '__main__':
    t: Series = data_subset_iloc(Series([1, 2]), [0])
    print(t)
