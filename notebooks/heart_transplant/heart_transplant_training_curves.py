import shelve

import argparse
import logging
from pandas import DataFrame, Series
from typing import List, Tuple, Iterable

from evaluation_functions import join_repeats_and_folds_cv_results, \
    compute_classification_metrics_from_results_with_statistics
from nested_cv import evaluate_method_on_sets, StaticHyperParameters, \
    get_cv_results_from_simple_cv_evaluation
from notebooks.heart_transplant.dependencies.heart_transplant_data import get_shuffled_cv_inputs_cached, \
    get_rolling_cv_inputs_cached
from notebooks.heart_transplant.dependencies.heart_transplant_functions import AgeGroup
from notebooks.heart_transplant.dependencies.heart_transplant_pipelines import get_xgboost_pipeline, \
    get_random_forest_pipeline
from utils import encode_dict_to_params

HEART_TRANSPLANT_TRAINING_CURVES_CV_SHUFFLED_IDENTIFIER = \
    'data/heart_transplant/heart_transplant_training_curves_cv'

HEART_TRANSPLANT_TRAINING_CURVES_EXPANDING_IDENTIFIER = \
    'data/heart_transplant/heart_transplant_training_curves_expanding'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('shuffled_cv', 'expanding'))
    parser.add_argument('--survival-days', type=int)
    parser.add_argument('--group', choices=('L_18', 'ME_18', 'ALL'))
    args = parser.parse_args()

    if args.type == 'shuffled_cv':
        evaluate(
            HEART_TRANSPLANT_TRAINING_CURVES_CV_SHUFFLED_IDENTIFIER,
            *get_shuffled_cv_inputs_cached()
        )
    elif args.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_TRAINING_CURVES_EXPANDING_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_rolling_cv_inputs_cached(
                survival_days=args.survival_days, group=AgeGroup[args.group]
            )
        )


# noinspection PyUnusedLocal
def evaluate(
    file_identifier: str, X: DataFrame, y: Series, X_valid: DataFrame, dataset_raw: DataFrame,
    sampling_sets: Iterable[Tuple[List[int], List[int]]]
):
    sampling_sets = list(sampling_sets)

    print('Dataset loaded')

    print('Starting training')

    logging.getLogger().setLevel(logging.DEBUG)

    methods = {
        'xgboost': lambda _X, _y: get_xgboost_pipeline(_X, _y).set_params(
            **encode_dict_to_params(
                {
                    'classifier': {
                        'colsample_bytree': 0.5338977067906682,
                        'gamma': 1.4443844778754271,
                        'learning_rate': 0.0453271241042913,
                        'max_depth': 3,
                        'min_child_weight': 8.0,
                        'n_estimators': 190,
                        'subsample': 0.38420232006551414
                    }
                }
            )
        ),
        'random_forest': lambda _X, _y: get_random_forest_pipeline(_X, _y).set_params(
            **encode_dict_to_params(
                {
                    'classifier': {
                        'bootstrap': True,
                        'max_depth': 15,
                        'max_features': 'auto',
                        'min_samples_leaf': 20,
                        'min_samples_split': 10,
                        'n_estimators': 447
                    }
                }
            )
        )
    }

    print('Omitted columns:', set(X.columns) - set(X_valid.columns))

    for method_name, pipeline in methods.items():
        print(method_name)
        feature_subset = list(X_valid.columns)

        with shelve.open(file_identifier) as data:
            if method_name in data:
                if len(data[method_name]) == len(feature_subset):
                    print('Skipping...')
                    continue
                else:
                    result_per_n_features = data[method_name]
            else:
                result_per_n_features = {}

        for n_of_features in [None, *reversed(range(1, len(feature_subset)))]:
            if n_of_features is None:
                feature_subset = list(X_valid.columns)
            else:
                # noinspection PyUnboundLocalVariable
                feature_subset = list(feature_importance.index)[:n_of_features]

            print(f'n features={len(feature_subset)}')

            if len(feature_subset) in result_per_n_features:
                continue

            X_with_feature_subset = X_valid[feature_subset]

            results = get_cv_results_from_simple_cv_evaluation(
                [
                    evaluate_method_on_sets(
                        lambda: pipeline(X_with_feature_subset, y),
                        X_with_feature_subset,
                        y,
                        StaticHyperParameters({}),
                        sampling_sets,
                        parallel=True,
                        n_jobs=len(sampling_sets),
                        feature_names=feature_subset,
                    )
                ]
            )

            output = {
                'features': feature_subset,
                'test': compute_classification_metrics_from_results_with_statistics(
                    y, results, threshold=0.5, target_variable='y_scores'
                ),
                'train': compute_classification_metrics_from_results_with_statistics(
                    y, results, threshold=0.5, target_variable='y_train_scores'
                ),
            }

            result_per_n_features[len(feature_subset)] = output

            with shelve.open(file_identifier) as s:
                s[method_name] = result_per_n_features

            feature_importance = join_repeats_and_folds_cv_results(results)['feature_importance']

            with shelve.open(file_identifier) as s:
                s[method_name] = result_per_n_features


if __name__ == '__main__':
    main()
