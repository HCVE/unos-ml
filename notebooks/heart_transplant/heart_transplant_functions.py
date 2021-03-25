from typing import Tuple, Dict, List, Optional, Union

import numpy as np
from pandas import DataFrame, Series, to_numeric
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from toolz import get

from functional import statements


class CustomLabelEncoder(LabelEncoder, TransformerMixin):

    def fit_transform(self, X, y=None, **fit_params):
        X_new = X.copy()
        for column in X.columns:
            X_new[column] = to_numeric(super().fit_transform(X[column]))

        return X_new

    @staticmethod
    def transform(X, _=None):
        X_new = X.copy()
        for column in X.columns:
            X_new[column] = to_numeric(super().transform(X[column]))

        return X_new


features_for_log_transform = (
    'tbili_don', 'age_don', 'tbili_don', 'age', 'distance', 'newpra', 'ischtime',
    'wgt_kg_tcr', 'most_rcnt_creat', 'creat_trr', 'tbili', 'bun_don',
    'creat_don', 'sgot_don', 'sgpt_don', 'tbili_don', 'wgt_kg_don_calc',
    'hgt_cm_calc', 'wgt_kg_calc', 'cmassratio'
)


def reverse_log_transform_row(row: Series) -> Series:
    new_row = row.copy()
    for log_plus_1_feature in features_for_log_transform:
        new_row[log_plus_1_feature] = np.exp(row[log_plus_1_feature]) - 1

    return new_row


def log_transform(dataset: DataFrame) -> DataFrame:
    new_dataset = dataset.copy()

    for log_plus_1_feature in features_for_log_transform:
        new_dataset[log_plus_1_feature] = np.log(dataset[log_plus_1_feature] + 1)

    return new_dataset


def get_features(metadata: Dict) -> List[str]:
    return list(metadata.keys())


def filter_out_unused_features(input_dataset: DataFrame, metadata: Dict) -> DataFrame:
    return input_dataset[get_features(metadata)]


def get_X_y_1_year_survival(input_dataset: DataFrame) -> Tuple[DataFrame, Series]:
    y = extract_y(input_dataset)
    defined_mask = ~y.isna()
    return (
        input_dataset[defined_mask].drop(
            ['death', 'deathr', 'futd', 'ptime', 'gtime', 'px_stat', 'pstatus', 'gstatus'],
            axis=1),
        y[defined_mask].astype(int)
    )


def extract_y(input_dataset):
    def get_y_from_row(row: Series) -> Optional[int]:
        inverse_log_futd = np.exp(row['futd'])

        if inverse_log_futd >= 365:
            return 0
        elif inverse_log_futd <= 365 and (row['death'] == 1):
            return 1
        else:
            return None

    # noinspection PyTypeChecker
    y = input_dataset.apply(get_y_from_row, axis=1)
    return y


def provide_long_labels(metadata: Dict, input_data: Union[Series, DataFrame]) -> Union[Series, DataFrame]:
    input_data_copy = input_data.copy()
    input_data_copy.index = input_data_copy.index.map(lambda index: f'{metadata[index]["name_long"]} [{index}]')
    return input_data_copy
