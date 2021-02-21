from typing import List, Tuple

import numpy
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.pipeline import Pipeline, make_pipeline
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from toolz import merge, concat
from xgboost import XGBClassifier

from functional import pipe
from nested_cv import BayesianOptimization
from utils import use_df, Debug
from methods.dummy_classifier import dummy_classifier


# noinspection PyUnusedLocal
def get_random_forest_pipeline(
        X: DataFrame, y: Series,
        balance_class: bool = True,
        n_jobs: int = 1
):
    return Pipeline([
        ('transform', get_transform(X)),
        ('scaler', StandardScaler()),
        ('classifier',
         RandomForestClassifier(n_jobs=n_jobs, **({'class_weight': 'balanced_subsample'} if balance_class else {})))
    ])


def get_xgboost_pipeline(
        X: DataFrame,
        y: Series,
        balance_class: bool = True,
        n_jobs: int = 12,
):
    class_counts = y.value_counts()
    class_ratio = class_counts[0] / class_counts[1]
    return Pipeline([
        ('transform', get_transform(X)),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            **merge(
                xgboost_base_hyperparameters,
                {'n_jobs': n_jobs},
                {'scale_pos_weight': class_ratio} if balance_class else {},
            ),
        ))
    ])


# noinspection PyUnusedLocal
def get_logistic_regression_pipeline(
        X: DataFrame, y: Series, features: List[str] = None, balance_class: bool = True,
        n_jobs: int = None
):
    return Pipeline([
        ('transform', get_transform(X)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty='l2', **({'class_weight': 'balanced'} if balance_class else {}))),
    ])


def get_dummy_classifier_pipeline(X: DataFrame, y: Series, features: List[str], n_jobs: int = None):
    return make_pipeline(DummyClassifier(strategy='stratified'))


def get_categorical_and_continuous_features(X: DataFrame) -> Tuple[List, List]:
    defined_categorical_features = [
        'func_stat_tcr', 'iabp_tcr', 'inotropes_tcr', 'ethcat', 'lvad ever', 'gender', 'gender_don', 'tah ever',
        'med_cond_trr', 'ecmo_trr', 'gstatus',
        'pstatus', 'ethcat_don', 'blood_inf_don', 'other_inf_don', 'pulm_inf_don', 'urine_inf_don', 'cod_cad_don',
        'death_mech_don', 'multiorg', 'abo_mat', 'lv_eject_meth', 'coronary_angio', 'vessels_50sten', 'biopsy_dgn',
        'ecd_donor', 'education', 'education', 'congenital', 'prior_card_surg_type_tcr', 'ventilator_tcr', 'rvad ever',
        'cancer_site_don',
    ]

    categorical_features = [
        feature_name for feature_name, series in X.items()
        if
        series.dtype == 'object' or feature_name in defined_categorical_features or feature_name in defined_categorical_features
    ]

    continuous_features = [
        feature_name for feature_name in X.columns if feature_name not in categorical_features
    ]

    return categorical_features, continuous_features


def get_pipelines():
    return {
        #     'xgboost_balanced': lambda features=None: make_pipeline(
        #             get_transform_pipeline(features=features),
        #             StandardScaler(),
        #             use_df(XGBClassifier)(
        #             eval_metric='logloss',
        #                 scale_pos_weight=class_ratio,
        #                 use_label_encoder=False,
        #                 n_jobs=12,
        #             )
        #         ),
        'xgboost': get_xgboost_pipeline,
        'dummy_classifier': get_dummy_classifier_pipeline,
        # 'random_forest': get_random_forest_pipeline,
        # 'ridge': get_logistic_regression_pipeline,
        #     ),
        #     'random_forest_na': partial(get_random_forest_pipeline, impute=False),
        #     'mlp': lambda features: make_pipeline(
        #         get_transform_pipeline(features=features),
        #         StandardScaler(),
        #         MLPClassifier(),
        #     ),
    }


def get_transform(
        X: DataFrame,
):
    used_categorical_features, used_continuous_features = get_final_feature_sets(X)
    keys = [list(numpy.sort([i for i in X[column].unique() if not (i != i)])) for column in used_categorical_features]
    return ColumnTransformer([
        *([(
            'categorical',
            make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OrdinalEncoder(categories=keys)
            ),
            used_categorical_features,
        )] if len(used_categorical_features) > 0 else []),
        *([(
            'continuous',
            SimpleImputer(strategy='mean'),
            used_continuous_features,
        )] if len(used_continuous_features) > 0 else []),
    ])


def get_final_feature_sets(X) -> Tuple[List[str], List[str]]:
    categorical_features, continuous_features = get_categorical_and_continuous_features(X)
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


xgboost_hyperopt = hp.choice(
    'base', [{
        'classifier': {
            'learning_rate': hp.uniform('learning_rate', 0.01, 1),
            'max_depth': scope.int(hp.quniform('max_depth', 2, 16, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
            'subsample': hp.uniform('subsample', 0.1, 1),
            'gamma': hp.uniform('gamma', 0, 5),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 5)),
        }
    }]
)

logistic_regression_hyperopt = hp.choice(
    'base', [
        {
            'classifier': {'C': hp.choice('classifier__C', [
                0.0001,
                0.001,
                0.1,
                1,
                10,
                100,
            ])}
        }
    ]
)

xgboost_base_hyperparameters = dict(
    eval_metric='logloss',
    use_label_encoder=False,
    tree_method='gpu_hist',
    gpu_id=0,
    n_jobs=26,
)
