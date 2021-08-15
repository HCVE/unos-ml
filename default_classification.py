import argparse
from datetime import time
from os.path import basename
from typing import List, Tuple

from filelock import FileLock
from pandas import DataFrame, Series

from include.contants import RANDOM_SEED
from include.data import get_shuffled_cv_inputs_cached, \
    get_rolling_cv_inputs_cached
from include.evaluation_functions import DefaultHyperParameters, evaluate_method_on_sets
from include.functional import pipe
from include.utils import evaluate_and_assign_if_not_present, LockedShelve, random_seed, \
    reverse_log_transform_dataset, AgeGroup, get_categorical_and_continuous_features, \
    get_categorical_and_continuous_features_one_hot
from pipelines import get_logistic_regression_pipeline, \
    get_random_forest_pipeline, get_xgboost_pipeline

HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_shuffled_cv'
HEART_TRANSPLANT_EXPANDING_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_expanding'


def main(arguments):
    random_seed(RANDOM_SEED)

    if arguments.type == 'shuffled_cv':
        evaluate(
            f'{HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER}_{arguments.survival_days}_{arguments.group}',
            *get_shuffled_cv_inputs_cached(
                survival_days=arguments.survival_days, group=AgeGroup[arguments.group]
            ),
            only_export_csv=arguments.only_export_csv,
            only_show_removed_columns=arguments.only_show_removed_columns
        )
    elif arguments.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_EXPANDING_IDENTIFIER}_{arguments.survival_days}_{arguments.group}',
            *get_rolling_cv_inputs_cached(
                survival_days=arguments.survival_days, group=AgeGroup[arguments.group]
            ),
            only_export_csv=arguments.only_export_csv,
            only_show_removed_columns=arguments.only_show_removed_columns
        )


def evaluate(
    file_identifier: str,
    X: DataFrame,
    y: Series,
    X_valid: DataFrame,
    dataset_raw: DataFrame,
    sampling_sets: List[Tuple[List[int], List[int]]],
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

    print(f'Starting training, n folds={len(sampling_sets)}')

    n_jobs = len(sampling_sets)
    common_pipeline_args = dict(
        memory=None,
        verbose=0,
    )
    parallel = True
    categorical_features, _ = get_categorical_and_continuous_features(X_valid)
    persistence = LockedShelve(file_identifier)

    method_name = 'random_forest_default'
    evaluate_and_assign_if_not_present(
        persistence,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_random_forest_pipeline(X_valid, n_jobs=1, **common_pipeline_args),
            X_valid,
            y,
            DefaultHyperParameters(),
            folds=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )
    method_name = 'l2_logistic_regression_default'
    evaluate_and_assign_if_not_present(
        persistence,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_logistic_regression_pipeline(
                X_valid,
                y,
                balance_class=True,
                _get_final_feature_sets=get_categorical_and_continuous_features_one_hot,
            ),
            X_valid,
            y,
            DefaultHyperParameters(),
            folds=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    xgboost_lock = FileLock("temporary/xgboost.lock")

    with xgboost_lock:
        method_name = 'xgboost_default'
        evaluate_and_assign_if_not_present(
            persistence,
            method_name,
            lambda: evaluate_method_on_sets(
                lambda: get_xgboost_pipeline(X_valid, y, **common_pipeline_args),
                X_valid,
                y,
                DefaultHyperParameters(),
                folds=sampling_sets,
                parallel=parallel,
                n_jobs=len(sampling_sets),
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
