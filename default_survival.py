import argparse
from os.path import basename
from typing import List, Tuple

from pandas import DataFrame, Series

from include.evaluation_functions import get_classification_metrics, evaluate_method_on_sets, \
    DefaultHyperParameters
from include.functional import pipe
from include.contants import RANDOM_SEED
from include.data import get_shuffled_cv_inputs_cached, \
    get_rolling_cv_inputs_cached
from pipelines import get_cox_ph_pipeline, \
    get_random_survival_forest_pipeline, get_survival_gradient_boosting_pipeline
from include.utils import evaluate_and_assign_if_not_present, LockedShelve, random_seed, \
    reverse_log_transform_dataset, AgeGroup, get_survival_y, get_categorical_and_continuous_features_one_hot

HEART_TRANSPLANT_SURVIVAL_CV_SHUFFLED_IDENTIFIER = 'data/heart_transplant/heart_transplant_survival_results_shuffled_cv'
HEART_TRANSPLANT_SURVIVAL_ROLLING_IDENTIFIER = 'data/heart_transplant/heart_transplant_survival_results_expanding'


def main():
    random_seed(RANDOM_SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('shuffled_cv', 'expanding', 'expanding_test'))
    parser.add_argument('--survival-days', type=int)
    parser.add_argument('--group', choices=('L_18', 'ME_18', 'ALL'))
    parser.add_argument('--only-export-csv', default=False, action='store_true')
    parser.add_argument('--only-show-removed-columns', default=False, action='store_true')
    args = parser.parse_args()

    if args.type == 'shuffled_cv':
        evaluate(
            f'{HEART_TRANSPLANT_SURVIVAL_CV_SHUFFLED_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_shuffled_cv_inputs_cached(
                survival_days=args.survival_days, group=AgeGroup[args.group]
            ),
            days=args.survival_days,
            only_export_csv=args.only_export_csv,
            only_show_removed_columns=args.only_show_removed_columns
        )
    elif args.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_SURVIVAL_ROLLING_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_rolling_cv_inputs_cached(
                survival_days=args.survival_days, group=AgeGroup[args.group]
            ),
            days=args.survival_days,
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
    days: int,
    only_export_csv: bool = False,
    only_show_removed_columns: bool = False,
):
    print('Dataset loaded')

    common_pipeline_args = dict()

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

    persistence = LockedShelve(file_identifier)

    print('Starting training')

    y_survival = get_survival_y(dataset_raw.loc[X_valid.index])

    def get_metrics(_, result):
        return get_classification_metrics(y, result)

    evaluate_and_assign_if_not_present(
        persistence,
        f'survival_gradient_boosting_default',
        lambda: evaluate_method_on_sets(
            lambda: get_survival_gradient_boosting_pipeline(
                X_valid, y_survival, days=days, n_jobs=2, balance_class=True, verbose=0
            ),
            X_valid,
            y_survival,
            DefaultHyperParameters(),
            folds=sampling_sets,
            parallel=True,
            n_jobs=len(sampling_sets),
            get_metrics=get_metrics,
            **common_pipeline_args,
        ),
    )

    evaluate_and_assign_if_not_present(
        persistence,
        f'survival_cox_ph_default',
        lambda: evaluate_method_on_sets(
            lambda: get_cox_ph_pipeline(
                X_valid,
                days=days,
                n_jobs=1,
                balance_class=True,
                verbose=0,
                _get_final_feature_sets=get_categorical_and_continuous_features_one_hot,
                **common_pipeline_args,
            ),
            X_valid,
            y_survival,
            DefaultHyperParameters(),
            folds=sampling_sets,
            n_jobs=len(sampling_sets),
            get_metrics=get_metrics,
        ),
    )

    evaluate_and_assign_if_not_present(
        persistence,
        f'survival_random_forest_default',
        lambda: evaluate_method_on_sets(
            lambda: get_random_survival_forest_pipeline(
                X_valid, y_survival, days=days, n_jobs=1, balance_class=True
            ),
            X_valid,
            y_survival,
            DefaultHyperParameters(),
            folds=sampling_sets,
            parallel=False,
            n_jobs=len(sampling_sets),
            get_metrics=get_metrics,
            **common_pipeline_args,
        ),
    )

    print('DONE')


if __name__ == '__main__':
    main()
