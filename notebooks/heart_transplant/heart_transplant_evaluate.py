import argparse
import logging
import shelve
from typing import List, Tuple, Iterable

from imblearn.pipeline import make_pipeline
from pandas import DataFrame, Series

from methods.random_forest.random_forest import RandomForestMethod
from nested_cv import evaluate_method_on_sets, BayesianOptimization, DefaultHyperParameters
from notebooks.heart_transplant.dependencies.heart_transplant_data import get_expanding_window_inputs_cached, \
    get_shuffled_cv_inputs_cached
from notebooks.heart_transplant.dependencies.heart_transplant_pipelines import get_logistic_regression_pipeline, \
    get_random_forest_pipeline, xgboost_hyperopt, get_final_features, get_xgboost_pipeline, logistic_regression_hyperopt
from utils import evaluate_and_assign_if_not_present

HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_shuffled_cv'
HEART_TRANSPLANT_EXPANDING_IDENTIFIER = 'data/heart_transplant/heart_transplant_results_expanding'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=('shuffled_cv', 'expanding'))
    parser.add_argument('--survival-days', type=int)
    parser.add_argument('--group', choices=('l_18', 'me_18', 'all'))
    args = parser.parse_args()

    if args.type == 'shuffled_cv':
        evaluate(HEART_TRANSPLANT_CV_SHUFFLED_IDENTIFIER, *get_shuffled_cv_inputs_cached())
    elif args.type == 'expanding':
        evaluate(
            f'{HEART_TRANSPLANT_EXPANDING_IDENTIFIER}_{args.survival_days}_{args.group}',
            *get_expanding_window_inputs_cached(survival_days=args.survival_days, group=args.group)
        )


def evaluate(
        file_identifier: str, X: DataFrame, y: Series, X_valid: DataFrame, dataset_raw: DataFrame,
        sampling_sets: Iterable[Tuple[List[int], List[int]]]
):
    sampling_sets = list(sampling_sets)

    print('Dataset loaded')

    results = shelve.open(file_identifier)

    print('Starting training')

    logging.getLogger().setLevel(logging.DEBUG)

    n_jobs = len(sampling_sets)
    evaluate_and_assign_if_not_present(
        results,
        f'ridge_default',
        lambda: evaluate_method_on_sets(
            lambda: get_logistic_regression_pipeline(X_valid, y, balance_class=True),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=True,
            n_jobs=n_jobs,
        ),
    )

    try:
        del results['ridge_default_oversampled']
    except KeyError:
        pass

    evaluate_and_assign_if_not_present(
        results,
        f'xgboost_default',
        lambda: evaluate_method_on_sets(
            lambda: get_xgboost_pipeline(X_valid, y),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=True,
            feature_names=get_final_features(X_valid),
            n_jobs=n_jobs,
        ),
    )

    results.sync()

    evaluate_and_assign_if_not_present(
        results,
        f'random_forest_default',
        lambda: evaluate_method_on_sets(
            lambda: make_pipeline(*get_random_forest_pipeline(X_valid, y, n_jobs=1)),
            X_valid,
            y,
            DefaultHyperParameters(),
            splits=sampling_sets,
            parallel=True,
            n_jobs=n_jobs,
        ),
    )

    results.sync()

    evaluate_and_assign_if_not_present(
        results,
        f'xgboost_optimized_roc',
        lambda: evaluate_method_on_sets(
            lambda: get_xgboost_pipeline(X_valid, y),
            X_valid,
            y,
            BayesianOptimization(xgboost_hyperopt, target_metric='roc_auc'),
            splits=sampling_sets,
            parallel=True,
            n_jobs=n_jobs,
        ),
    )

    results.sync()

    evaluate_and_assign_if_not_present(
        results,
        f'ridge_optimized_roc',
        lambda: evaluate_method_on_sets(
            lambda: get_logistic_regression_pipeline(X_valid, y, balance_class=True),
            X_valid,
            y,
            BayesianOptimization(logistic_regression_hyperopt),
            splits=sampling_sets,
            parallel=True,
            n_jobs=n_jobs,
        ),
    )

    results.sync()

    evaluate_and_assign_if_not_present(
        results,
        f'random_forest_optimized_roc',
        lambda: evaluate_method_on_sets(
            lambda: get_random_forest_pipeline(X_valid, y, balance_class=True, n_jobs=-1),
            X_valid,
            y,
            BayesianOptimization(RandomForestMethod.get_hyperopt_space(), target_metric='roc_auc'),
            splits=sampling_sets,
            parallel=True,
            n_jobs=10,
        ),
    )
    results.sync()

    # X_expert_features = get_feature_subset(EXPERTISE_BASED_FEATURES, X)
    #
    # evaluate_and_assign_if_not_present(
    #     results,
    #     f'ridge_optimized_roc_expert_features',
    #     lambda: evaluate_method_on_sets(
    #         lambda: get_logistic_regression_pipeline(X_expert_features, y, balance_class=True),
    #         X_expert_features,
    #         y,
    #         BayesianOptimization(logistic_regression_hyperopt, iterations=10, target_metric='roc_auc'),
    #         splits=sampling_sets,
    #         parallel=True,
    #         n_jobs=n_jobs,
    #     ),
    # )
    #
    # results.sync()
    #
    # evaluate_and_assign_if_not_present(
    #     results,
    #     f'xgboost_optimized_roc_expert_features',
    #     lambda: evaluate_method_on_sets(
    #         lambda: get_xgboost_pipeline(X_expert_features, y, n_jobs=5),
    #         X_expert_features,
    #         y,
    #         BayesianOptimization(xgboost_hyperopt, target_metric='roc_auc'),
    #         splits=sampling_sets,
    #         parallel=True,
    #         n_jobs=n_jobs,
    #     ),
    # )
    #
    # results.sync()
    #
    # evaluate_and_assign_if_not_present(
    #     results,
    #     f'random_forest_optimized_roc_expert_features',
    #     lambda: evaluate_method_on_sets(
    #         lambda: get_random_forest_pipeline(X_expert_features, y, n_jobs=1),
    #         X_expert_features,
    #         y,
    #         BayesianOptimization(RandomForestMethod.get_hyperopt_space(), target_metric='roc_auc'),
    #         splits=sampling_sets,
    #         parallel=True,
    #         n_jobs=n_jobs,
    #     ),
    # )
    #
    # results.sync()
    #
    # evaluate_and_assign_if_not_present(
    #     results,
    #     f'xgboost_static_optimized',
    #     lambda: evaluate_method_on_sets(
    #         lambda: get_xgboost_pipeline(X_expert_features, y),
    #         X_expert_features,
    #         y,
    #         StaticHyperParameters({
    #             'classifier__colsample_bytree': 0.5080817028881132,
    #             'classifier__gamma': 3.092864665001854,
    #             'classifier__learning_rate': 0.08951636759116892,
    #             'classifier__max_depth': 2,
    #             'classifier__min_child_weight': 4.0,
    #             'classifier__n_estimators': 140,
    #             'classifier__subsample': 0.7437975255362691
    #         }),
    #         splits=sampling_sets,
    #         parallel=False,
    #         n_jobs=10,
    #     ),
    # )

    results.close()


if __name__ == '__main__':
    main()
