from typing import List, Tuple, Any, Callable

from hyperopt import hp
from hyperopt.pyll import scope
from joblib import Memory
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from toolz import merge
from xgboost import XGBClassifier

from dependencies.heart_transplant_functions import get_final_feature_sets, \
    remove_column_prefix, SurvivalWrapper
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from dependencies.utils import DFPassthrough
from wrapped_sklearn import DFColumnTransformer, DFSimpleImputer, DFOneHotEncoder, DFStandardScaler, DFPipeline, \
    DFLogisticRegression, DFOrdinalEncoder

GetFinalFeatures = Callable[[DataFrame], Tuple[List[str], List[str]]]


def get_random_forest_pipeline(
    X: DataFrame,
    balance_class: bool = True,
    n_jobs: int = 1,
    verbose: int = None,
    memory: Memory = None,
) -> Pipeline:
    return Pipeline(
        [
            *get_preprocessing(X),
            (
                'classifier',
                RandomForestClassifier(
                    n_jobs=n_jobs,
                    **({
                        'class_weight': 'balanced_subsample'
                    } if balance_class else {})
                )
            )
        ],
        memory=memory,
        verbose=verbose,
    )


random_forest_hyperopt = hp.choice(
    'base', [
        {
            'classifier': {
                'bootstrap': hp.choice('classifier_bootstrap', [True, False]),
                'max_depth': scope.int(hp.quniform('max_depth', 2, 100, 1)),
                'max_features': hp.choice('classifier_max_features', ['auto', 'log2', None]),
                'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
                'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
                'n_estimators': scope.int(hp.uniform('classifier__n_estimators', 5, 500)),
            }
        }
    ]
)


def get_xgboost_pipeline(
    X: DataFrame,
    y: Series,
    balance_class: bool = True,
    n_jobs: int = 1,
    memory: Memory = None,
    verbose: int = None,
    impute: bool = False,
) -> Pipeline:
    class_counts = y.value_counts()
    class_ratio = class_counts[0] / class_counts[1]
    return Pipeline(
        [
            *get_preprocessing(X, impute=impute),
            (
                'classifier',
                XGBClassifier(
                    **merge(
                        dict(
                            eval_metric='logloss',
                            use_label_encoder=False,
                            tree_method='gpu_hist',
                            gpu_id=0,
                            n_jobs=1,
                        ),
                        {'n_jobs': n_jobs},
                        {'scale_pos_weight': class_ratio} if balance_class else {},
                    ),
                )
            )
        ],
        memory=memory,
        verbose=verbose,
    )


xgboost_hyperopt = hp.choice(
    'base', [
        {
            'classifier': {
                'learning_rate': hp.uniform('learning_rate', 0.01, 1),
                'max_depth': scope.int(hp.quniform('max_depth', 2, 16, 1)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
                'subsample': hp.uniform('subsample', 0.1, 1),
                'gamma': hp.uniform('gamma', 0, 5),
                'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 5)),
            }
        }
    ]
)


# noinspection PyUnusedLocal
def get_logistic_regression_pipeline(
    X: DataFrame,
    y: Series,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    do_one_hot: bool = True,
    memory: Any = None,
    verbose: int = None,
    _get_final_feature_sets: GetFinalFeatures = get_final_feature_sets
):
    return Pipeline(
        [
            *get_preprocessing(
                X, do_one_hot=do_one_hot, _get_final_feature_sets=_get_final_feature_sets
            ),
            (
                'classifier',
                DFLogisticRegression(
                    penalty='l2', **({
                        'class_weight': 'balanced'
                    } if balance_class else {})
                )
            ),
        ],
        memory=memory,
        verbose=verbose,
    )


logistic_regression_hyperopt = hp.choice(
    'base', [{
        'classifier': {
            'C': hp.loguniform('classifier__C', -3, 3)
        }
    }]
)


class CoxPHSurvivalAnalysisDays(SurvivalWrapper, CoxPHSurvivalAnalysis):
    ...


cox_ph_hyperopt = hp.choice(
    'base', [{
        'classifier': {
            'alpha': hp.loguniform('classifier__alpha', -3, 3)
        }
    }]
)


# noinspection PyAbstractClass
class RandomSurvivalForest_(SurvivalWrapper, RandomSurvivalForest):
    ...


class GradientBoostingSurvivalAnalysis_(SurvivalWrapper, GradientBoostingSurvivalAnalysis):
    ...


# noinspection PyUnusedLocal
def get_random_survival_forest_pipeline(
    X: DataFrame,
    y: Any,
    days: int,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    memory: Any = None,
    verbose: int = None,
):
    # noinspection PyArgumentList
    return Pipeline(
        [
            *get_preprocessing(X),
            (
                'classifier',
                RandomSurvivalForest_(
                    verbose=2,
                    n_jobs=n_jobs,
                    n_estimators=60,
                    max_depth=50,
                ).set_survival_days(days)
            ),
        ],
        memory=memory,
        verbose=verbose,
    )


def get_survival_gradient_boosting_pipeline(
    X: DataFrame,
    y: Any,
    days: int,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    memory: Any = None,
    verbose: int = None,
):
    used_categorical_features, used_continuous_features = get_final_feature_sets(X)

    return Pipeline(
        [
            *get_preprocessing(X),
            ('classifier', GradientBoostingSurvivalAnalysis_(verbose=2).set_survival_days(days)),
        ],
        memory=memory,
        verbose=verbose,
    )


def get_cox_ph_pipeline(
    X: DataFrame,
    days: int,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    memory: Any = None,
    verbose: int = 0,
    _get_final_feature_sets=get_final_feature_sets,
):
    return DFPipeline(
        [
            *get_preprocessing(X, do_one_hot=True, _get_final_feature_sets=_get_final_feature_sets),
            (
                'classifier', CoxPHSurvivalAnalysisDays(
                    alpha=0.5,
                    verbose=verbose,
                ).set_survival_days(days)
            ),
        ],
        memory=memory,
        verbose=verbose,
    )


def get_preprocessing(
    X: DataFrame,
    do_one_hot: bool = False,
    impute: bool = True,
    _get_final_feature_sets: GetFinalFeatures = get_final_feature_sets
) -> List:
    used_categorical_features, used_continuous_features = _get_final_feature_sets(X)
    return [
        (
            'preprocess',
            DFColumnTransformer(
                [
                    (
                        'categorical',
                        DFPipeline(
                            [
                                (
                                    'c1',
                                    'passthrough'
                                    if not impute else DFSimpleImputer(strategy='most_frequent'),
                                ),
                                (
                                    'c2',
                                    'passthrough' if do_one_hot else DFOrdinalEncoder(),
                                ),
                                (
                                    'c3',
                                    DFOneHotEncoder(
                                        cols=used_categorical_features,
                                        use_cat_names=True,
                                    ) if do_one_hot else DFPassthrough(),
                                ),
                            ]
                        ) if len(used_categorical_features) > 0 else DFPassthrough(),
                        used_categorical_features,
                    ),
                    (
                        'continuous',
                        DFPipeline(
                            [
                                (
                                    'c1', 'passthrough'
                                    if not impute else DFSimpleImputer(strategy='mean')
                                ),
                                ('c2', DFStandardScaler()),
                            ]
                        ) if len(used_continuous_features) > 0 else 'passthrough',
                        used_continuous_features,
                    ),
                ]
            ),
        ),
        (
            'remove_prefix',
            FunctionTransformer(remove_column_prefix),
        ),
    ]
