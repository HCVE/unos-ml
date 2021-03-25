import shelve
from os.path import basename

import argparse
import logging
import numpy
from datetime import time
from pandas import DataFrame, Series
from typing import List, Tuple

from cache import get_memory
from functional import pipe
from methods.random_forest.random_forest import RandomForestMethod
from nested_cv import evaluate_method_on_sets, BayesianOptimization, DefaultHyperParameters
from notebooks.heart_transplant.dependencies.heart_transplant_data import get_shuffled_cv_inputs_cached, \
    get_expanding_window_inputs_for_test_cached, get_rolling_cv_inputs_cached
from notebooks.heart_transplant.dependencies.heart_transplant_functions import reverse_log_transform_dataset, \
    AgeGroup
from notebooks.heart_transplant.dependencies.heart_transplant_pipelines import get_logistic_regression_pipeline, \
    get_random_forest_pipeline, xgboost_hyperopt, get_final_features, get_xgboost_pipeline, \
    logistic_regression_hyperopt, get_categorical_and_continuous_features
from utils import evaluate_and_assign_if_not_present

HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_shuffled_cv'
HEART_TRANSPLANT_EXPANDING_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_expanding'


def main():
    numpy.random.seed(49788)

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('shuffled_cv', 'expanding', 'expanding_test'))
    parser.add_argument('--survival-days', type=int)
    parser.add_argument('--group', choices=('L_18', 'ME_18', 'ALL'))
    parser.add_argument('--only-export-csv', default=False, action='store_true')
    parser.add_argument('--only-show-removed-columns', default=False, action='store_true')
    args = parser.parse_args()

    if args.type == 'shuffled_cv':
        evaluate(
            HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER,
            *get_shuffled_cv_inputs_cached(),
            only_export_csv=args.only_export_csv,
            only_show_removed_columns=args.only_show_removed_columns
        )
    elif args.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_EXPANDING_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_rolling_cv_inputs_cached(
                survival_days=args.survival_days, group=AgeGroup[args.group]
            ),
            only_export_csv=args.only_export_csv,
            only_show_removed_columns=args.only_show_removed_columns
        )
    elif args.type == 'expanding_test':
        evaluate(
            f'{HEART_TRANSPLANT_EXPANDING_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_expanding_window_inputs_for_test_cached(
                survival_days=args.survival_days, group=AgeGroup(args.group)
            ),
            only_export_csv=args.only_export_csv,
            only_show_removed_columns=args.only_show_removed_columns
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

    def get_shelve():
        return shelve.open(file_identifier)

    print(f'Starting training, n folds={len(sampling_sets)}')

    logging.getLogger().setLevel(logging.DEBUG)

    n_jobs = len(sampling_sets)

    common_pipeline_args = dict(memory=None, verbose=0)
    parallel = False

    categorical_features, _ = get_categorical_and_continuous_features(X_valid)

    method_name = 'l2_logistic_regression_default'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_logistic_regression_pipeline(X_valid, y, **common_pipeline_args),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    method_name = 'xgboost_default'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_xgboost_pipeline(X_valid, y, **common_pipeline_args),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=parallel,
            feature_names=get_final_features(X_valid),
            n_jobs=len(sampling_sets),
        ),
    )

    method_name = 'random_forest_default'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_random_forest_pipeline(X_valid, y, n_jobs=1, **common_pipeline_args),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    method_name = 'xgboost_tuned'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_xgboost_pipeline(
                X_valid,
                y,
            ),
            X_valid,
            y,
            BayesianOptimization(xgboost_hyperopt, target_metric='roc_auc', iterations=50),
            splits=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    method_name = 'random_forest_tuned'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_random_forest_pipeline(X_valid, y, n_jobs=2),
            X_valid,
            y,
            BayesianOptimization(
                RandomForestMethod.get_hyperopt_space(),
                target_metric='roc_auc',
                iterations=50,
            ),
            splits=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )

    method_name = 'l2_logistic_regression_tuned'
    evaluate_and_assign_if_not_present(
        get_shelve,
        method_name,
        lambda: evaluate_method_on_sets(
            lambda: get_logistic_regression_pipeline(
                X_valid,
                y,
                balance_class=True,
            ),
            X_valid,
            y,
            BayesianOptimization(logistic_regression_hyperopt, iterations=10),
            splits=sampling_sets,
            parallel=parallel,
            n_jobs=n_jobs,
        ),
    )



if __name__ == '__main__':
    main()
