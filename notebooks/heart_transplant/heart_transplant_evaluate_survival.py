import shelve
from os.path import basename

import argparse
import logging
from pandas import DataFrame, Series
from typing import List, Tuple

from evaluation_functions import get_classification_metrics, get_train_test_sampling
from functional import pipe
from nested_cv import evaluate_method_on_sets, DefaultHyperParameters
from notebooks.heart_transplant.dependencies.heart_transplant_data import get_shuffled_cv_inputs_cached, \
    get_expanding_window_inputs_for_test_cached, get_rolling_cv_inputs_cached
from notebooks.heart_transplant.dependencies.heart_transplant_functions import reverse_log_transform_dataset, \
    get_survival_y, AgeGroup
from notebooks.heart_transplant.dependencies.heart_transplant_pipelines import get_cox_ph_pipeline
from utils import evaluate_and_assign_if_not_present

HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_shuffled_cv'
HEART_TRANSPLANT_EXPANDING_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_expanding2'


def main():
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
                survival_days=args.survival_days, group=AgeGroup(args.group)
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

    print('Starting training')

    logging.getLogger().setLevel(logging.DEBUG)

    y_survival = get_survival_y(dataset_raw)

    evaluate_and_assign_if_not_present(
        get_shelve,
        f'survival_cox_ph_default',
        lambda: evaluate_method_on_sets(
            lambda: get_cox_ph_pipeline(X_valid, y_survival, n_jobs=1, balance_class=True),
            X_valid,
            y_survival,
            DefaultHyperParameters(),
            splits=get_train_test_sampling(X_valid),
            parallel=False,
            n_jobs=20,
            get_metrics=lambda _, results: get_classification_metrics(y, results)
        ),
        force_execute=True,
    )


if __name__ == '__main__':
    main()
