import logging
import re
from enum import Enum
from functools import partial
from numbers import Number
from typing import Tuple, Dict, List, Union, Iterable, Set, Mapping, TypedDict, Optional

import numpy
import numpy as np
import pandas
# noinspection PyUnresolvedReferences
import swifter
from matplotlib import pyplot
from pandas import DataFrame, Series
from rpy2 import robjects
from rpy2.robjects.packages import importr
from sklearn.model_selection import KFold
from toolz import concat

from cache import memory
from custom_types import CVSampling, NestedCV, NestedTrainSet, ValueWithStatistics
from functional import pipe
from dependencies.heart_transplant_constants import FEATURES_TO_DROP
from dependencies.utils import balanced_range, get_ilocs_by_callback, capitalize_first, map_columns, remove_prefix

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
    # noinspection PyTypeChecker
    return np.log(number + 1)


def get_features(metadata: Dict) -> List[str]:
    return [key for key in metadata.keys() if not metadata[key].get('remove', False)]


def filter_out_unused_features(input_dataset: DataFrame, metadata: Dict) -> DataFrame:
    return input_dataset[get_features(metadata)].drop(columns=FEATURES_TO_DROP)


def get_X_y_1_year_survival(
    X: DataFrame,
    dataset_raw: DataFrame,
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
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


def format_feature(
    metadata: Dict,
    feature_name: str,
    remove_donor_and_recipient=True,
    note_is_log: bool = True
) -> str:
    is_log = feature_name.startswith("log_")
    base_name = feature_name[len('log_'):] if is_log else feature_name
    decomposed_feature_name = decompose_one_hot_feature_name(base_name, metadata)
    try:
        metadata_item = metadata[decomposed_feature_name['name']]
        if remove_donor_and_recipient:
            final_base_name = pipe(
                metadata_item['name_long'],
                lambda s: re.sub(' ?recipient', '', s, flags=re.IGNORECASE),
                lambda s: re.sub(' ?donor', '', s, flags=re.IGNORECASE) if 'ECCT' not in s else s,
                str.strip,
                capitalize_first,
            )

        try:
            decomposed_value = metadata_item['value_map'][float(decomposed_feature_name["value"])]
        except (KeyError, TypeError, ValueError):
            decomposed_value = decomposed_feature_name["value"]

        return final_base_name + \
               (f' • {decomposed_value}' if decomposed_feature_name['value'] else '') + \
               ((' (log)' if is_log else '') if note_is_log else '')
    except KeyError:
        return feature_name


def format_features(metadata: Dict, features: Iterable[str]) -> List[str]:
    return [format_feature(metadata, feature) for feature in features]


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


def extract_y_from_preprocessed(input_dataset, survival_days: int = SURVIVAL_DAYS_DEFAULT):
    # noinspection PyTypeChecker
    y = input_dataset.swifter.apply(
        partial(get_y_from_preprocessed_row, survival_days=survival_days), axis=1
    )
    return y


def extract_y_from_raw(input_dataset, survival_days: int = SURVIVAL_DAYS_DEFAULT):
    # noinspection PyTypeChecker
    y = input_dataset.swifter.apply(
        partial(get_y_from_raw_row, survival_days=survival_days), axis=1
    )
    return y


def get_y_from_preprocessed_row(
    row: Series,
    survival_days: int = SURVIVAL_DAYS_DEFAULT,
) -> Union[int, float]:
    numpy.seterr(all='raise')
    row_raw = reverse_log_transform_row(row)
    return get_y_from_raw_row(row_raw, survival_days)


def get_y_from_raw_row(row_raw, survival_days):
    if row_raw['futd'] > survival_days:
        return 0
    elif row_raw['futd'] <= survival_days and (row_raw['death'] == 1):
        return 1
    else:
        return np.nan


def provide_long_labels(metadata: Dict, input_data: Union[Series,
                                                          DataFrame]) -> Union[Series, DataFrame]:
    input_data_copy = input_data.copy()
    input_data_copy.index = input_data_copy.index.map(
        lambda index: f'{metadata[index]["name_long"]} [{index}]'
    )
    return input_data_copy


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

    year_start = year_start if year_start is not None else (
        data_frame['tx_year'].min() + minimum_training_years + validation_size_years_int +
        test_size_years - 1
    )

    year_stop = year_stop if year_stop is not None else data_frame['tx_year'].max()

    if n_windows is None:
        n_windows = (year_stop - year_start) + 1

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
                            lambda row: (
                                threshold - test_size_years - validation_size_years_int < row[
                                    'tx_year'] <= threshold - test_size_years
                            ), data_frame
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

    year_start = year_start if year_start is not None else (
        data_frame['tx_year'].min() + minimum_training_years + validation_size_years_int +
        test_size_years - 1
    )

    year_stop = year_stop if year_stop is not None else data_frame['tx_year'].max()

    if n_windows is None:
        n_windows = (year_stop - year_start) + 1

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
                        lambda row:
                        (threshold - test_size_years < row['tx_year'] <= threshold), data_frame
                    )
                )
            ),
        )

    return result


get_rolling_cv_cached = memory.cache(get_rolling_cv)


def remove_missing_columns(X: DataFrame, sampling_sets: Iterable, verbose: int = 0) -> DataFrame:
    removed_columns: Set = set()
    for train_set, test_set in sampling_sets:
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
    L_18 = 'L_18',
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


def format_heart_transplant_method_name(identifier: str) -> str:
    output = ""
    if 'survival_random_forest' in identifier:
        output = 'Random Survival Forest'
    elif 'random_forest' in identifier:
        output = 'Random Forest'
    elif 'survival_gradient_boosting' in identifier:
        output = 'Survival Gradient Boosting'
    elif 'l2_logistic_regression' in identifier:
        output = "L2 Logistic Regression"
    elif 'xgboost' in identifier:
        output = 'XGBoost'
    elif 'cox' in identifier:
        output = 'Cox regression'

    if 'no_imputation' in identifier:
        output += ' No Impute'

    return output


def format_heart_transplant_method_name_tuned(identifier: str) -> str:
    return r"$\bf{optimised}$ " + format_heart_transplant_method_name(identifier) + ' in rolling CV'


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
        ),
    )


def get_feature_metadata(feature_name: str, metadata: Mapping) -> Mapping:
    feature_name_base = feature_name[4:] if feature_name.startswith('log_') else feature_name
    return metadata[feature_name_base]


def get_rolling_cv_callback(_X: DataFrame, _dataset_raw: DataFrame) -> CVSampling:
    return list(
        get_rolling_cv_cached(
            _X.assign(tx_year=_dataset_raw['tx_year']),
            n_windows=None,
            test_size_years=1,
            minimum_training_years=10,
            year_stop=2016
        )
    )


def get_shuffled_10_fold_callback(_X: DataFrame) -> CVSampling:
    return list(KFold(n_splits=10, shuffle=True).split(_X))


class DecomposedOneHotFeatureName(TypedDict):
    name: str
    value: Optional[str]


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


def compute_auc_proc(y_score: Series, y_true: Series):
    y_true = y_true.loc[y_score.index]
    pROC = importr('pROC')
    roc_r = pROC.roc(robjects.FloatVector(y_true), robjects.FloatVector(y_score))
    ci = pROC.ci(roc_r)
    return ValueWithStatistics(
        pROC.auc(roc_r)[0],
        ci=(ci[0], ci[2]),
        std=None,
    )


def set_style(ax=None) -> None:
    if ax is None:
        ax = pyplot.gca()
    pyplot.rcParams['axes.titlepad'] = 10
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_fontsize(13)
    ax.xaxis.labelpad = 5
    ax.yaxis.label.set_fontsize(13)
    ax.yaxis.labelpad = 7


def get_final_feature_sets(X) -> Tuple[List[str], List[str]]:
    categorical_features, continuous_features = get_categorical_and_continuous_features(X)
    return (
        [feature for feature in categorical_features if feature in X.columns],
        [feature for feature in continuous_features if feature in X.columns],
    )


def get_final_feature_sets_for_one_hot(X) -> Tuple[List[str], List[str]]:
    categorical_features, continuous_features = get_final_feature_sets(X)
    return (
        [feature for feature in categorical_features if feature in X.columns],
        [feature for feature in continuous_features if feature in X.columns],
    )


def get_final_features(X) -> List[str]:
    return pipe(
        get_final_feature_sets(X),
        concat,
        list,
    )


def remove_column_prefix(X: DataFrame) -> DataFrame:
    return map_columns(
        lambda column_name: pipe(
            column_name,
            partial(remove_prefix, 'categorical__'),
            partial(remove_prefix, 'continuous__'),
        ),
        X,
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
        feature_name for feature_name, series in X.items()
        if series.dtype == 'object' or feature_name in defined_categorical_features
    ]

    continuous_features = [
        feature_name for feature_name in X.columns if feature_name not in categorical_features
    ]

    return categorical_features, continuous_features


def get_categorical_and_continuous_features_one_hot(X: DataFrame) -> Tuple[List, List]:
    features = ['diag', 'func_stat_trr']
    categorical_features, continuous_features = get_categorical_and_continuous_features(X)
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
