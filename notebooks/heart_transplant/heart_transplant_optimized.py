from functools import partial

from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cache import save_data, load_data, memory
from methods.xgboost.xgboost_model import XGBoostMethod
from nested_cv import evaluate_method_on_sets, BayesianOptimization, DefaultHyperParameters
from notebooks.heart_transplant.dependencies.heart_transplant_data import get_reduced_binary_dataset_cached, \
    get_reduced_binary_dataset
from notebooks.heart_transplant.dependencies.heart_transplant_functions import get_expanding_windows, \
    remove_missing_columns
from notebooks.heart_transplant.dependencies.heart_transplant_pipelines import get_transform_pipeline_no_df, \
    get_logistic_regression_pipeline
from utils import evaluate_and_assign_if_not_present, get_class_ratio

HEART_TRANSPLANT_OPTIMIZED_RESULTS_IDENTIFIER = 'heart_transplant/heart_transplant_results'


def get_inputs():
    X, y, dataset_raw = get_reduced_binary_dataset()
    sampling_sets = list(get_expanding_windows(
        X.assign(tx_year=dataset_raw['tx_year']),
        n_windows=10,
        test_size_years=1,
        minimum_training_years=10,
        year_stop=2015
    ))

    X_valid = remove_missing_columns(X, sampling_sets, verbose=1)

    return X, y, X_valid, dataset_raw, sampling_sets


get_inputs_cached = memory.cache(get_inputs)

if __name__ == '__main__':

    print('Dataset loaded')

    X, y, X_valid, dataset_raw, sampling_sets = get_inputs_cached()

    print('Preprocessed')

    try:
        results = load_data(HEART_TRANSPLANT_OPTIMIZED_RESULTS_IDENTIFIER)
    except FileNotFoundError:
        results = {}

    print('Starting training')

    evaluate_and_assign_if_not_present(
        results,
        'xgboost_expanding_f1',
        lambda: evaluate_method_on_sets(
            lambda: Pipeline([
                ('transform', get_transform_pipeline_no_df(X_valid, features=list(X_valid.columns))),
                ('scaler', StandardScaler()),
                # ('classifier', RandomForestClassifier(
                #     # eval_metric='logloss',
                #     # scale_pos_weight=get_class_rati,
                #     # use_label_encoder=False,
                #     n_jobs=1,
                # )),
                ('classifier', XGBClassifier(
                    eval_metric='logloss',
                    # scale_pos_weight=get_class_rati,
                    use_label_encoder=False,
                    n_jobs=26,
                )),
            ]),
            X_valid,
            y,
            BayesianOptimization(hp.choice(
                'base', [{
                    'classifier': {
                        'learning_rate': hp.uniform('learning_rate', 0.01, 1),
                        'max_depth': scope.int(hp.quniform('max_depth', 2, 16, 1)),
                        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
                        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
                        'subsample': hp.uniform('subsample', 0.5, 1),
                        'gamma': hp.uniform('gamma', 0, 5),
                        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 1)),
                    }
                }]
            ), target_metric='f1'),
            splits=sampling_sets,
            n_jobs=10,
        )
    )

    evaluate_and_assign_if_not_present(
        results,
        'xgboost_roc_balanced',
        lambda: evaluate_method_on_sets(
            lambda: Pipeline([
                ('transform', get_transform_pipeline_no_df(X_valid, features=list(X_valid.columns))),
                ('scaler', StandardScaler()),
                # ('classifier', RandomForestClassifier(
                #     # eval_metric='logloss',
                #     # scale_pos_weight=get_class_rati,
                #     # use_label_encoder=False,
                #     n_jobs=1,
                # )),
                ('classifier', XGBClassifier(
                    scale_pos_weight=get_class_ratio(y),
                    eval_metric='logloss',
                    use_label_encoder=False,
                    tree_method='gpu_hist',
                    gpu_id=0,
                    n_jobs=26,
                )),
            ]),
            X_valid,
            y,
            BayesianOptimization(hp.choice(
                'base', [{
                    'classifier': {
                        'learning_rate': hp.uniform('learning_rate', 0.01, 1),
                        'max_depth': scope.int(hp.quniform('max_depth', 2, 16, 1)),
                        'min_child_weight': hp.quniform('min_child_weight', 1, 8, 1),
                        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.8),
                        'subsample': hp.uniform('subsample', 0.5, 1),
                        'gamma': hp.uniform('gamma', 0, 5),
                        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 200, 1)),
                    }
                }]
            ), target_metric='roc_auc'),
            splits=sampling_sets,
            parallel=False,
            n_jobs=10,
        ),
        was_not_present_callback=lambda whole_dict: save_data(HEART_TRANSPLANT_OPTIMIZED_RESULTS_IDENTIFIER, whole_dict)
    )

    evaluate_and_assign_if_not_present(
        results,
        'ridge_logistic_regression',
        lambda: evaluate_method_on_sets(
            partial(get_logistic_regression_pipeline, X_valid, y),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=True,
            n_jobs=10,
        ),
        was_not_present_callback=lambda whole_dict: save_data(HEART_TRANSPLANT_OPTIMIZED_RESULTS_IDENTIFIER, whole_dict)
    )
