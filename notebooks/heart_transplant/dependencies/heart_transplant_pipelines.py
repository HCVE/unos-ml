from category_encoders.ordinal import OrdinalEncoder
from functools import partial
from hyperopt import hp
from hyperopt.pyll import scope
from imblearn.pipeline import Pipeline
from joblib import Memory
from pandas import DataFrame, Series
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from toolz import merge, concat
from typing import List, Tuple, Any
from xgboost import XGBClassifier

from functional import pipe
from sksurv.linear_model import CoxPHSurvivalAnalysis
from utils import map_columns, \
    remove_prefix
from wrapped_sklearn import DFColumnTransformer, DFSimpleImputer, DFOneHotEncoder, DFStandardScaler, DFPipeline, \
    DFLogisticRegression


# noinspection PyUnusedLocal
def get_random_forest_pipeline(
    X: DataFrame,
    y: Series,
    balance_class: bool = True,
    n_jobs: int = 1,
    verbose: int = None,
    memory: Memory = None,
):
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


def get_xgboost_pipeline(
    X: DataFrame,
    y: Series,
    balance_class: bool = True,
    n_jobs: int = 1,
    memory: Memory = None,
    verbose: int = None,
):
    class_counts = y.value_counts()
    class_ratio = class_counts[0] / class_counts[1]
    return Pipeline(
        [
            *get_preprocessing(X),
            (
                'classifier',
                XGBClassifier(
                    **merge(
                        xgboost_base_hyperparameters,
                        {'n_jobs': n_jobs},
                        {'scale_pos_weight': class_ratio} if balance_class else {},
                    ),
                )
            )
        ],
        memory=memory,
        verbose=verbose,
    )


xgboost_base_hyperparameters = dict(
    eval_metric='logloss',
    use_label_encoder=False,
    tree_method='gpu_hist',
    gpu_id=0,
    n_jobs=26,
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


def get_logistic_regression_pipeline(
    X: DataFrame,
    y: Series,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    do_one_hot: bool = True,
    memory: Any = None,
    verbose: int = None,
):
    return Pipeline(
        [
            *get_preprocessing(X, do_one_hot=do_one_hot),
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
    'base',
    [{
        'classifier': {
            'C': hp.choice('classifier__C', [
                0.0001,
                0.001,
                0.1,
                1,
                10,
                100,
                1000,
            ])
        }
    }]
)


class CoxPHSurvivalAnalysis1Year(CoxPHSurvivalAnalysis):

    def predict_proba(self, X) -> DataFrame:
        y_score_0 = [fn(365) for fn in self.predict_survival_function(X)]
        y_score_1 = [1 - y_score for y_score in y_score_0]
        return DataFrame(
            {
                'y_predict_probabilities_0': y_score_0,
                'y_predict_probabilities_1': y_score_1
            },
            index=X.index
        )


cox_ph_hyperopt = hp.choice(
    'base', [
        {
            'classifier': {
                'alpha': hp.choice('classifier__alpha', [
                    0.0001,
                    0.001,
                    0.1,
                    1,
                    10,
                    100,
                    1000,
                ])
            }
        }
    ]
)


# noinspection PyUnusedLocal
def get_cox_ph_pipeline(
    X: DataFrame,
    features: List[str] = None,
    balance_class: bool = True,
    n_jobs: int = None,
    memory: Any = None,
    verbose: int = 0,
):
    used_categorical_features, used_continuous_features = get_final_feature_sets(X)

    return DFPipeline(
        [
            *get_preprocessing(X, do_one_hot=True),
            ('classifier', CoxPHSurvivalAnalysis1Year(alpha=0.5)),
        ],
        memory=memory,
        verbose=verbose,
    )


def get_dummy_classifier_pipeline(X: DataFrame, y: Series, features: List[str], n_jobs: int = None):
    return make_pipeline(DummyClassifier(strategy='stratified'))


def get_preprocessing(X: DataFrame, do_one_hot: bool = False) -> Tuple:
    used_categorical_features, used_continuous_features = get_final_feature_sets(X)
    return (
        (
            'imputer',
            DFColumnTransformer(
                [
                    (
                        'categorical',
                        DFPipeline(
                            [
                                ('c1', DFSimpleImputer(strategy='most_frequent')),
                                ('c2', OrdinalEncoder()),
                            ]
                        ) if len(used_categorical_features) > 0 else 'passthrough',
                        used_categorical_features,
                    ),
                    (
                        'continuous',
                        DFPipeline(
                            [
                                ('c1', DFSimpleImputer(strategy='mean')),
                                ('c2', DFStandardScaler()),
                            ]
                        ) if len(used_continuous_features) > 0 else 'passthrough',
                        used_continuous_features,
                    ),
                ]
            )
        ),
        (
            'remove_prefix',
            FunctionTransformer(remove_column_prefix),
        ),
        (
            'onehot',
            DFColumnTransformer(
                [
                    *(
                        [
                            (
                                'categorical',
                                DFOneHotEncoder(handle_unknown='ignore'),
                                used_categorical_features,
                            )
                        ] if len(used_categorical_features) > 0 else []
                    ),
                ],
                remainder='passthrough',
            ) if do_one_hot else 'passthrough',
        ),
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
        'func_stat_tcr',
        'iabp_tcr',
        'inotropes_tcr',
        'ethcat',
        'lvad ever',
        'gender',
        'gender_don',
        'tah ever',
        'med_cond_trr',
        'ecmo',
        'ecmo_trr',
        'gstatus',
        'pstatus',
        'ethcat_don',
        'blood_inf_don',
        'other_inf_don',
        'pulm_inf_don',
        'urine_inf_don',
        'cod_cad_don',
        'death_mech_don',
        'multiorg',
        'abo_mat',
        'lv_eject_meth',
        'coronary_angio',
        'vessels_50sten',
        'biopsy_dgn',
        'ecd_donor',
        'education',
        'education',
        'congenital',
        'prior_card_surg_type_tcr',
        'ventilator_tcr',
        'rvad ever',
        'cancer_site_don',
    ]

    categorical_features = [
        feature_name for feature_name, series in X.items()
        if series.dtype == 'object' or feature_name in defined_categorical_features
    ]

    continuous_features = [
        feature_name for feature_name in X.columns if feature_name not in categorical_features
    ]

    return categorical_features, continuous_features


# noinspection PyTypeChecker
def get_transform(X: DataFrame, ):
    used_categorical_features, used_continuous_features = get_final_feature_sets(X)
    return DFColumnTransformer(
        [(
            'categorical',
            OrdinalEncoder(handle_missing='return_nan'),
            used_categorical_features,
        )],
        remainder='passthrough'
    )


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
