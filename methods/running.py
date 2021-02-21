from functools import partial
from typing import List

import pandas
from toolz.curried import map

from api.api_utils import structure_feature_importance
from arguments import get_params
from custom_types import SupervisedPayload, CrossValidationType, ResultPayload, ClassificationMetricsWithStatistics
from evaluation_functions import cross_validate_model, ModelCVResult, \
    compute_ci_for_metrics_collection, join_folds_cv_result, ModelResult, compute_curves_folds, get_1_class_y_score, \
    compute_classification_metrics_from_result
from functional import flatten, pipe
from methods.methods_utils import start_loading, make_model, get_result_for_comparison, stop_loading
from nested_cv import evaluate_repeated_nested_cross_validation
from utils import object2dict


async def run_generic(
    get_pipeline,
    get_parameter_space,
    output,
    payload: SupervisedPayload,
    tab_key,
    parallel=True,
    fast=None,
):
    fast = fast if fast is not None else get_params('fast')
    await output(start_loading(tab_key))

    X, y_true, classifier = make_model(
        get_pipeline,
        payload['features'],
        payload['label'],
        payload['configuration'].get('pipeline'),
        payload['configuration'].get('reduce_classes'),
        path=payload['dataset']['file']
    )
    metrics_with_ci: ClassificationMetricsWithStatistics
    results: ResultPayload
    if payload['cross_validation_type'] == CrossValidationType.NESTED.value:
        repeat_results: List[ModelCVResult] = evaluate_repeated_nested_cross_validation(
            get_pipeline, get_parameter_space, payload
        )
        metrics_for_folds = pipe(
            repeat_results,
            map(lambda repeat_result: compute_classification_metrics_from_result(y_true, repeat_result)),
            flatten,
            list,
        )
        metrics_with_ci = compute_classification_metrics_from_result(metrics_for_folds)
        feature_importance = pipe(
            [repeat_result['feature_importance'] for repeat_result in repeat_results],
            partial(pandas.concat, axis=1),
            partial(pandas.DataFrame.mean, axis=1),
            structure_feature_importance,
        )
        results = ResultPayload(metrics=metrics_with_ci, feature_importance=feature_importance)
    else:
        model_cv_result: ModelCVResult = cross_validate_model(
            X, y_true, classifier, fast=fast, parallel=parallel
        )
        model_result: ModelResult = join_folds_cv_result(model_cv_result)

        result_for_comparison, y_for_comparison = get_result_for_comparison(
            payload['configuration']['reduce_classes'], model_cv_result, y_true, payload['label']
        )
        metrics_for_folds = compute_classification_metrics_from_result(y_for_comparison, result_for_comparison)
        metrics_with_ci = compute_ci_for_metrics_collection(metrics_for_folds)

        results = ResultPayload(
            metrics=metrics_with_ci,
            feature_importance=structure_feature_importance(
                model_result['feature_importance']['mean']
            ),
        )
        y_scores_single = [get_1_class_y_score(y_score) for y_score in model_cv_result['y_scores']]
        curves = compute_curves_folds(y_scores_single, y_true)

    tab_data = object2dict(
        {
            'results': results,
            'roc': {
                'tpr': curves.curve_vertical_roc,
                'fpr': curves.curve_horizontal,
            },
        }
    )
    await output(stop_loading(tab_key, tab_data))
