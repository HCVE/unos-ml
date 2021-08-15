import argparse
import logging
from datetime import time
from os.path import basename

from filelock import FileLock
from pandas import DataFrame, Series

from include.custom_types import NestedCV
from include.evaluation_functions import BayesianOptimization, evaluate_method_on_sets_nested, \
    get_classification_metrics
from include.functional import pipe
from include.contants import RANDOM_SEED
from include.data import get_nested_rolling_cv_inputs, \
    get_nested_shuffled_cv_inputs
from pipelines import get_logistic_regression_pipeline, \
    get_random_forest_pipeline, xgboost_hyperopt, get_xgboost_pipeline, \
    logistic_regression_hyperopt, get_cox_ph_pipeline, cox_ph_hyperopt, random_forest_hyperopt
from include.utils import evaluate_and_assign_if_not_present, LockedShelve, random_seed, \
    reverse_log_transform_dataset, AgeGroup, get_survival_y, get_categorical_and_continuous_features, \
    get_categorical_and_continuous_features_one_hot

HEART_TRANSPLANT_CV_SHUFFLED_NESTED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_nested_shuffled_cv'
HEART_TRANSPLANT_EXPANDING_NESTED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results__nested_expanding'


def main(arguments):
    random_seed(RANDOM_SEED)

    if arguments.type == 'shuffled_cv':
        evaluate(
            f'{HEART_TRANSPLANT_CV_SHUFFLED_NESTED_IDENTIFIER}_{arguments.survival_days}_{arguments.group}',
            *get_nested_shuffled_cv_inputs(
                survival_days=arguments.survival_days, group=AgeGroup[arguments.group]
            ),
            days=arguments.survival_days,
            only_export_csv=arguments.only_export_csv,
            only_show_removed_columns=arguments.only_show_removed_columns,
        )
    elif arguments.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_EXPANDING_NESTED_IDENTIFIER}_{arguments.survival_days}_{arguments.group}',
            *get_nested_rolling_cv_inputs(
                survival_days=arguments.survival_days, group=AgeGroup[arguments.group]
            ),
            days=arguments.survival_days,
            only_export_csv=arguments.only_export_csv,
            only_show_removed_columns=arguments.only_show_removed_columns,
        )


def evaluate(
    file_identifier: str,
    X: DataFrame,
    y: Series,
    X_valid: DataFrame,
    dataset_raw: DataFrame,
    sampling_sets: NestedCV,
    days: int,
    only_export_csv: bool = False,
    only_show_removed_columns: bool = False,
):
    print(time())

    print('Dataset loaded')

    if only_show_removed_columns:
        print(pipe(
            set(X) - set(X_valid),
            list,
            sorted,
        ))
        return

    if only_export_csv:
        print('Reverse log...')
        X_for_export = reverse_log_transform_dataset(X_valid)
        print('Exporting...')
        X_for_export.assign(y=y, tx_year=dataset_raw['tx_year']).to_csv(
            f'./data/heart_transplant/csv/{basename(file_identifier)}.csv', index=False
        )
        return

    n_folds = len(list(sampling_sets))
    n_jobs = n_folds
    logging.getLogger().setLevel(logging.DEBUG)
    common_pipeline_args = dict()
    parallel = True
    categorical_features, _ = get_categorical_and_continuous_features(X_valid)
    persistence = LockedShelve(file_identifier)

    print(f'Starting training, n folds={n_folds}')

    method_name = 'l2_logistic_regression_tuned'
    evaluate_and_assign_if_not_present(
        persistence,
        method_name,
        lambda: evaluate_method_on_sets_nested(
            lambda: get_logistic_regression_pipeline(
                X_valid,
                y,
                balance_class=True,
                _get_final_feature_sets=get_categorical_and_continuous_features_one_hot,
                **common_pipeline_args,
            ),
            X_valid,
            y,
            BayesianOptimization(logistic_regression_hyperopt, iterations=10),
            folds=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    method_name = 'random_forest_tuned'
    evaluate_and_assign_if_not_present(
        persistence,
        method_name,
        lambda: evaluate_method_on_sets_nested(
            lambda: get_random_forest_pipeline(
                X_valid,
                n_jobs=2,
                **common_pipeline_args,
            ),
            X_valid,
            y,
            BayesianOptimization(
                random_forest_hyperopt,
                target_metric='roc_auc',
                iterations=50,
            ),
            folds=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    y_survival = get_survival_y(dataset_raw.loc[X_valid.index])

    evaluate_and_assign_if_not_present(
        persistence,
        key='survival_cox_ph_tuned',
        callback=lambda: evaluate_method_on_sets_nested(
            lambda: get_cox_ph_pipeline(
                X_valid,
                days,
                **common_pipeline_args,
            ),
            X_valid,
            y_survival,
            BayesianOptimization(cox_ph_hyperopt, iterations=10),
            folds=sampling_sets,
            parallel=True,
            n_jobs=len(sampling_sets),
            get_metrics=lambda _, result: get_classification_metrics(y, result)
        ),
    )

    xgboost_lock = FileLock("temporary/xgboost.lock")

    with xgboost_lock:
        method_name = 'xgboost_tuned'
        evaluate_and_assign_if_not_present(
            persistence,
            method_name,
            lambda: evaluate_method_on_sets_nested(
                lambda: get_xgboost_pipeline(
                    X_valid,
                    y,
                    n_jobs=1,
                    **common_pipeline_args,
                ),
                X_valid,
                y,
                BayesianOptimization(xgboost_hyperopt, target_metric='roc_auc', iterations=50),
                folds=sampling_sets,
                parallel=True,
                n_jobs=n_jobs,
            ),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('shuffled_cv', 'expanding', 'expanding_test'))
    parser.add_argument('--survival-days', type=int)
    parser.add_argument('--group', choices=('L_18', 'ME_18', 'ALL'))
    parser.add_argument('--only-export-csv', default=False, action='store_true')
    parser.add_argument('--only-show-removed-columns', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
