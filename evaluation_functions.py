import logging
from dataclasses import dataclass
from functools import partial, reduce
from multiprocessing.pool import Pool
from statistics import mean, stdev, StatisticsError
# noinspection Mypy
from typing import Iterable, Optional, Any, Dict, Union, TypedDict, Callable, Tuple, TypeVar
from typing import List

import numpy as np
import pandas
import tabulate
from PyALE import ale
from functional_pipeline import pipeline, flatten
from matplotlib import pyplot
from numpy import NaN, float64
from pandas import DataFrame, concat
from pandas import Series
from pandas.core.indexing import IndexingError
from scipy.interpolate import interp1d
from six import moves
from sklearn import clone
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score, \
    precision_recall_curve, roc_curve, f1_score, average_precision_score, balanced_accuracy_score, \
    explained_variance_score, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import export_text
from toolz import curry, identity
from toolz.curried import get, pluck, map, filter, valmap

from cache import memory
from utils import object2dict
from custom_types import Estimator, ClassificationMetrics, ClassificationMetricsWithSTD, ValueWithStatistics, \
    ClassificationMetricsWithStatistics, \
    DataStructure, ConfusionMatrix, ConfusionMatrixWithStatistics, Method, RegressionMetrics, RegressionMetricsWithSTD, \
    RegressionMetricsWithStatistics
from formatting import dict_to_table_horizontal, format_method, tabulate_formatted, format_structure, format_decimal, \
    format_metric_short
from functional import flatten, statements, find_index_right
from functional import pass_args, mapl, pipe, decorate_unpack, find_index, unzip, add_index
from statistics_functions import confidence_interval, get_dof
from utils import get_object_attributes, ll, get_log_level, log, Timer

DEFAULT_THRESHOLD = 0.5
T1 = TypeVar('T1')


@dataclass
class ModelExecutionResult(DataStructure):
    y_train_predicts: List[Series]
    y_predicts: List[Series]
    y_train_scores: List[Series]
    y_scores: List[Series]
    feature_importance: Series
    models: List[Any]
    elapsed: float


class FoldModelExecutionResult(TypedDict):
    y_predict_probabilities: Series
    y_predict: Series
    y_train_predict: Series
    y_train_predict_probabilities: Series
    feature_importance: Series
    model: Estimator
    elapsed: float


@dataclass
class ModelExecutionResultVector(DataStructure):
    y_train_predict: Series
    y_predict: Series
    y_train_score: Series
    y_score: Series
    feature_importance: Series
    model: Any
    elapsed: float


@dataclass
class ModelCVResult(DataStructure):
    y_train_predicts: List[Series]
    y_predicts: List[Series]
    y_train_scores: List[DataFrame]
    y_scores: List[DataFrame]
    feature_importance: List[Series]
    models: List[Any]
    elapsed: float


class ModelResult(TypedDict):
    y_test_score: DataFrame
    y_test_predict: Series
    y_train_predict: Series
    y_train_score: DataFrame
    feature_importance: Union[Series, DataFrame]
    model: Estimator
    elapsed: float


def compute_regression_metrics(y_score, y_true) -> RegressionMetrics:
    y_true_masked = y_true.loc[y_score.index]
    return RegressionMetrics(
        explained_variance=explained_variance_score(y_true_masked, y_score),
        r2=r2_score(y_true_masked, y_score),
        mean_absolute_error=mean_absolute_error(y_true_masked, y_score),
    )


def compute_classification_metrics(
        y_score, y_true, threshold: float = DEFAULT_THRESHOLD, ignore_warning: bool = False
) -> ClassificationMetrics:
    y_predict = y_score >= threshold
    y_true_masked = y_true.loc[y_predict.index]
    roc = roc_curve(y_true_masked, y_score)
    fpr, tpr = get_roc_point_by_threshold(threshold, *roc)
    npv = get_metrics_from_confusion_matrix(
        get_confusion_from_threshold(y_true_masked, y_score, threshold)
    ).npv

    precision = precision_score(y_true_masked, y_predict, **({'zero_division': 0} if ignore_warning else {}))

    return ClassificationMetrics(
        recall=tpr,
        precision=precision,
        balanced_accuracy=balanced_accuracy_score(y_true_masked, y_predict),
        f1=f1_score(y_true_masked, y_predict),
        tnr=1 - fpr,
        fpr=fpr,
        fnr=1 - tpr,
        average_precision=average_precision_score(y_true_masked, y_score),
        accuracy=accuracy_score(y_true_masked, y_predict),
        roc_auc=roc_auc_score(y_true_masked, y_score),
        npv=npv,
    )


def compute_classification_metrics_folds(
        y_scores: List[Series],
        y: Series,
        threshold: float = DEFAULT_THRESHOLD,
) -> Optional[ClassificationMetricsWithSTD]:
    return pipeline(
        y_scores,
        [
            map(lambda y_score: compute_classification_metrics(get_1_class_y_score(y_score), y, threshold)),
            list,
            average_list_dicts_metric_std,
        ],
    )


def compute_regression_metrics_folds(
        y_scores: List[Series],
        y: Series,
) -> Optional[ClassificationMetricsWithSTD]:
    return pipeline(
        y_scores,
        [
            map(lambda y_score: compute_regression_metrics(y_score, y)),
            list,
            average_list_dicts,
            lambda item: RegressionMetrics(**item),
        ],
    )


def average_list_dicts(metrics: List[Dict]) -> Optional[Dict]:
    if len(metrics) == 0:
        return None
    output = {}
    try:
        keys = metrics[0].__dict__.keys()
    except AttributeError:
        keys = metrics[0].keys()

    for key in keys:
        values = list(
            map(lambda item: getattr(item, key) if hasattr(item, key) else item[key], metrics)
        )
        mean_value = mean(values)
        try:
            stdev_value = stdev(values)
        except StatisticsError:
            stdev_value = 0

        output[key] = (mean_value, stdev_value)
    return output


def average_list_dicts_metric_std(metrics: Union[Any]) -> Optional[ClassificationMetricsWithSTD]:
    d = average_list_dicts(metrics)
    if d:
        return ClassificationMetricsWithSTD(**d)
    else:
        return None


def execute_model_predict_proba(classifier, X):
    return classifier.predict_proba(X)


def cross_validate_model(
        X,
        y,
        classifier,
        cv=10,
        fast=False,
        reduce=False,
        parallel=True,
        predict_proba=None,
        n_jobs=12,
        return_model: bool = True
) -> ModelCVResult:
    predict_proba = predict_proba or execute_model_predict_proba
    if not fast:
        cv = StratifiedKFold(n_splits=cv)
        sets = cv.split(X, y)
    else:
        if reduce:
            X = X[:100]
            y = y[:100]
        amount = round(len(X) * (1 / cv))
        sets = [(X.index[:-amount], X.index[-amount:])]

    # TODO: dirty hack
    try:
        classifier.set_params(onehot=None)
    except ValueError:
        pass

    return cross_validate_model_sets(
        classifier, X, y, sets, predict_proba, parallel, n_jobs, return_model
    )


cross_validate_model_cached = memory.cache(cross_validate_model)


class WorkerInput(TypedDict):
    X_train: DataFrame
    y_train: Series
    X_test: DataFrame
    classifier: Estimator
    predict_proba: Callable[[Estimator, DataFrame], Series]
    feature_names: Optional[List[str]]
    return_model: bool


def cross_validate_model_sets(
        classifier,
        X,
        y,
        sets,
        predict_proba=execute_model_predict_proba,
        parallel=True,
        n_jobs=12,
        return_model: bool = True,
        filter_X_test: Callable[[DataFrame], DataFrame] = identity,
        feature_names: Optional[List[str]] = None,
) -> ModelCVResult:
    worker_input: List[WorkerInput] = [
        WorkerInput(
            X_train=X.iloc[train],
            y_train=y.iloc[train],
            X_test=filter_X_test(X.iloc[test]),
            classifier=classifier,
            predict_proba=predict_proba,
            return_model=return_model,
            feature_names=feature_names,
        ) for (train, test) in sets
    ]
    if parallel:
        with Pool(min(len(worker_input), n_jobs)) as p:
            result = p.map(cross_validate_model_fold, worker_input)
    else:
        result = list(map(cross_validate_model_fold, worker_input))

    return result_from_fold_results(result)


cross_validate_model_sets_cached = memory.cache(cross_validate_model_sets, ignore=['n_jobs'])


def cross_validate_model_sets_args(
        get_x_y, n_jobs=12, parallel=True, *args, **kwargs
) -> ModelCVResult:
    X, y = get_x_y()
    return cross_validate_model_sets(X=X, y=y, n_jobs=n_jobs, parallel=parallel, *args, **kwargs)


cross_validate_model_sets_args_cached = memory.cache(
    cross_validate_model_sets_args, ignore=['n_jobs', 'parallel']
)


def cross_validate_model_fold(chunk_input: WorkerInput) -> ModelResult:
    log("Execution fold", level=2)
    timer = Timer()
    classifier = clone(chunk_input['classifier'])
    X_train = chunk_input['X_train']
    y_train = chunk_input['y_train']
    X_test = chunk_input['X_test']
    return_model = chunk_input['return_model']

    feature_names = \
        chunk_input['feature_names'] if \
            ('feature_names' in chunk_input and chunk_input['feature_names'] is not None) \
            else list(X_train.columns)

    classifier.fit(
        X_train,
        y_train,
    )

    y_predict = Series(classifier.predict(X_test), index=X_test.index)
    y_train_predict = Series(classifier.predict(X_train), index=X_train.index)

    try:
        y_predict_probabilities_raw = classifier.predict_proba(X_test)
        y_train_predict_probabilities_raw = classifier.predict_proba(X_train)
    except AttributeError:
        y_predict_probabilities = y_predict
        y_train_predict_probabilities = y_train_predict
    else:
        probability_columns = [
            f'y_predict_probabilities_{i}' for i in range(y_predict_probabilities_raw.shape[1])
        ]
        y_predict_probabilities = DataFrame(
            y_predict_probabilities_raw, index=X_test.index, columns=probability_columns
        )
        y_train_predict_probabilities = DataFrame(
            y_train_predict_probabilities_raw, index=X_train.index, columns=probability_columns
        )

    if y_predict.dtype == np.float:
        y_predict = y_predict \
            .map(lambda v: 0 if v < 0 else v) \
            .map(lambda v: 1 if v > 1 else v) \
            .map(lambda v: round(v))

    try:
        feature_importance = Series(
            classifier[-1].feature_importances_,
            index=feature_names,
        )
    except (TypeError, AttributeError):
        try:
            classifier[-1].coef_
        except AttributeError:
            feature_importance = None
            logging.debug("No feature importance in the result")
        else:
            feature_importance = Series(classifier[-1].coef_[0], index=feature_names)

    if not return_model:
        try:
            classifier[-1].get_booster().__del__()
        except AttributeError:
            pass

    if get_log_level() == 1:
        print(".")
    else:
        log("Execution fold finished", level=2)

    return ModelResult(
        y_test_score=y_predict_probabilities,
        y_test_predict=y_predict,
        y_train_predict=y_train_predict,
        y_train_score=y_train_predict_probabilities,
        feature_importance=feature_importance,
        model=classifier[-1] if return_model else None,
        elapsed=timer.elapsed_cpu()
    )


cross_validate_model_fold_cached = memory.cache(cross_validate_model_fold)


def cross_validate_model_fold_args(
        classifier, get_x_y, train_index, test_index, return_model: bool = True
) -> ModelResult:
    X, y = get_x_y()
    return cross_validate_model_fold(
        WorkerInput(
            classifier=classifier,
            X_train=X.iloc[train_index],
            y_train=y.iloc[train_index],
            X_test=X.iloc[test_index],
            return_model=return_model,
            predict_proba=None,
        )
    )


cross_validate_model_fold_args_cached = memory.cache(cross_validate_model_fold_args)


def result_from_fold_results(results: Iterable[ModelResult]) -> ModelCVResult:
    results = list(results)

    return ModelCVResult(
        feature_importance=pipeline(
            results,
            [
                map(get('feature_importance')),
                list,
            ],
        ),
        y_scores=pipeline(
            results,
            [
                map(get('y_test_score')),
                list,
            ],
        ),
        y_train_scores=pipeline(
            results,
            [
                map(get('y_train_score')),
                list,
            ],
        ),
        y_predicts=pipeline(
            results,
            [
                map(get('y_test_predict')),
                list,
            ],
        ),
        y_train_predicts=pipeline(
            results,
            [
                map(get('y_train_predict')),
                list,
            ],
        ),
        models=pipeline(
            results,
            [
                map(get('model')),
                list,
            ],
        ),
        elapsed=pipeline(results, [
            map(get('elapsed')),
            max,
        ])
    )


def join_repeats_and_folds_cv_results(results: List[ModelCVResult]) -> ModelResult:
    return ModelResult(**pipe(
        results,
        join_repeats_cv_results,
        join_folds_cv_result,
    ))


def join_repeats_cv_results(results: List[ModelCVResult]) -> ModelCVResult:
    return reduce(
        lambda result1, result2: ModelCVResult(
            y_train_predicts=[*result1['y_train_predicts'], *result2['y_train_predicts']],
            y_predicts=[*result1['y_predicts'], *result2['y_predicts']],
            y_train_scores=[*result1['y_train_scores'], *result2['y_train_scores']],
            y_scores=[*result1['y_scores'], *result2['y_scores']],
            feature_importance=[*result1['feature_importance'], *result2['feature_importance']],
            models=[*result1['models'], *result2['models']],
            elapsed=result1['elapsed'] + result2['elapsed'],
        ),
        results,
    )


def get_feature_importance_from_cv_result(result: ModelCVResult) -> DataFrame:
    return statements(
        feature_importance_vector := pandas.concat(
            result['feature_importance'],
            axis=1
        ).transpose(),
        DataFrame(
            {
                'mean': feature_importance_vector.mean(),
                'std': feature_importance_vector.std(),
            }
        ).sort_values(by='mean', ascending=False, inplace=False)
    )


def join_folds_cv_result(result: ModelCVResult) -> ModelResult:
    return ModelResult(
        feature_importance=get_feature_importance_from_cv_result(result) if result['feature_importance'][
                                                                                0] is not None else None,
        y_test_score=pandas.concat(result['y_scores']).sort_index(),
        y_test_predict=pandas.concat(result['y_predicts']).sort_index(),
        y_train_predict=pandas.concat(result['y_train_predicts']).sort_index(),
        y_train_score=pandas.concat(result['y_train_scores']).sort_index(),
        models=list(flatten(result['models'])),
        elapsed=result['elapsed'],
    )


def get_result_vector_from_result(result: ModelCVResult) -> ModelResult:
    result_single = join_folds_cv_result(result)
    try:
        single_vector_y_test_score = result_single['y_test_score'].iloc[:, 1]
    except IndexingError:
        pass
    else:
        result_single['y_test_score'] = single_vector_y_test_score

    return result_single


def get_full_vector_result_comparison(y, result: ModelResult) -> DataFrame:
    return pandas.concat(
        [
            pandas.DataFrame(
                {
                    'actual': y[result['y_test_predict'].index],
                    'predict_class': result['y_test_predict'],
                }
            ), result['y_test_score']
        ],
        axis=1,
    ).sort_values(['actual', 'predict_class'], ascending=False)


@dataclass
class ModelResultCurves:
    curve_horizontal: List[float]
    curve_vertical_recall_precision: List[float]
    curve_vertical_roc: List[float]


@dataclass
class ModelResultCurvesStd(ModelResultCurves):
    curve_vertical_recall_precision_std: List[float]
    curve_vertical_roc_std: List[float]


curves_interpolate_default = 100


def compute_curves(
        y_score: Series, y_true: Series, interpolate=curves_interpolate_default
) -> ModelResultCurves:
    y_masked = y_true[y_score.index]
    fpr, tpr, _ = roc_curve(
        y_masked,
        y_score,
    )

    curve_horizontal = np.linspace(0, 1, interpolate)

    precision, recall, _ = precision_recall_curve(
        y_masked,
        y_score,
    )
    interpolation_recall_precision = interp1d(recall, precision, assume_sorted=False)
    curve_vertical_recall_precision = interpolation_recall_precision(curve_horizontal)
    interpolation_roc = interp1d(fpr, tpr)
    curve_vertical_roc = interpolation_roc(curve_horizontal)

    return ModelResultCurves(
        curve_horizontal,
        curve_vertical_recall_precision,
        curve_vertical_roc,
    )


def compute_curves_folds(
        y_score_folds: List[Series],
        y_true: Series,
        interpolate=curves_interpolate_default
) -> ModelResultCurvesStd:
    curves_folds = []
    curve_horizontal: List[float] = []
    for y_score in y_score_folds:
        curves = compute_curves(y_score, y_true, interpolate)
        curve_horizontal = curves.curve_horizontal
        curves_folds.append(curves)

    curve_vertical_recall_precision_aligned = zip(
        *map(lambda i: i.curve_vertical_recall_precision, curves_folds)
    )
    curve_vertical_recall_precision_mean = list(map(mean, curve_vertical_recall_precision_aligned))
    curve_vertical_recall_precision_std = list(map(stdev, curve_vertical_recall_precision_aligned))

    curve_vertical_roc_aligned = zip(*map(lambda i: i.curve_vertical_roc, curves_folds))
    curve_vertical_roc_mean = list(map(mean, curve_vertical_roc_aligned))
    curve_vertical_roc_std = list(map(stdev, curve_vertical_roc_aligned))

    return ModelResultCurvesStd(
        curve_horizontal=curve_horizontal,
        curve_vertical_recall_precision=curve_vertical_recall_precision_mean,
        curve_vertical_recall_precision_std=curve_vertical_recall_precision_std,
        curve_vertical_roc=curve_vertical_roc_mean,
        curve_vertical_roc_std=curve_vertical_roc_std,
    )


def compute_regression_metrics_from_result(
        y: Series,
        result: ModelCVResult,
) -> Optional[List[RegressionMetrics]]:
    return [
        compute_regression_metrics(y_score, y)
        for y_score in result['y_scores']
    ]


def compute_classification_metrics_from_result(
        y: Series,
        result: ModelCVResult,
        target_variable: str = 'y_scores',
        threshold: float = DEFAULT_THRESHOLD,
        ignore_warning: bool = False,
) -> Optional[List[ClassificationMetrics]]:
    return [
        compute_classification_metrics(get_1_class_y_score(score), y, threshold=threshold, ignore_warning=ignore_warning)
        for score in result[target_variable]
    ]


def get_classification_metrics(
        y: Series,
        result: ModelCVResult,
) -> Optional[ClassificationMetricsWithSTD]:
    return compute_classification_metrics_folds(result['y_scores'], y)


@curry
def get_regression_metrics(
        y: Series,
        result: ModelCVResult,
) -> Optional[ClassificationMetricsWithSTD]:
    return compute_regression_metrics_folds(result['y_scores'], y)


@curry
def report_cross_validation(y: Series, result: ModelCVResult) -> None:
    metrics = get_classification_metrics(y, result)
    if metrics:
        print(dict_to_table_horizontal(metrics))


def compute_ci_for_metrics_collection(metrics: List[ClassificationMetrics]) -> Dict:
    attributes = get_object_attributes(metrics[0])
    metrics_with_ci_dict = {
        attribute: pass_args(
            confidence_interval(get_dof(), list(pluck(attribute, metrics))),
            lambda m, ci, std: ValueWithStatistics(m, std, ci),
        )
        for attribute in attributes
    }
    return metrics_with_ci_dict


def get_best_threshold_from_roc(
        tps: np.array,
        fps: np.array,
        thresholds: np.array,
) -> Tuple[float, int]:
    J = np.abs(tps - fps)
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh, ix


def get_best_threshold_from_results(y_true: Series, results: List[ModelCVResult]) -> float:
    fpr, tpr, thresholds = compute_threshold_averaged_roc(y_true, results)
    best_threshold, index = get_best_threshold_from_roc(tpr, fpr, thresholds)
    return best_threshold


def compute_classification_metrics_from_results_with_statistics(
        y_true: Series,
        results: List[ModelCVResult],
        threshold: Optional[float] = None,
        target_variable: str = 'y_scores',
        ignore_warning: bool = False,
) -> ClassificationMetricsWithStatistics:
    chosen_threshold = threshold if threshold is not None else get_best_threshold_from_results(
        y_true, results
    )
    return pipeline(
        results,
        [
            partial(mapl, partial(
                compute_classification_metrics_from_result,
                y_true,
                threshold=chosen_threshold,
                target_variable=target_variable,
                ignore_warning=ignore_warning,
            )),
            flatten,
            list,
            compute_ci_for_metrics_collection,
        ],
    )


def compute_regression_metrics_from_results_with_statistics(
        y_true: Series,
        results: List[ModelCVResult],
) -> RegressionMetricsWithStatistics:
    return pipeline(
        results,
        [
            partial(mapl, partial(compute_regression_metrics_from_result, y_true)),
            flatten,
            list,
            compute_ci_for_metrics_collection,
            lambda item: RegressionMetricsWithStatistics(**item)
        ],
    )


def compute_metrics_from_result_ci(
        y_true: Series,
        result: ModelCVResult,
        threshold: Optional[float] = None
) -> ClassificationMetricsWithStatistics:
    chosen_threshold = threshold if threshold is not None else get_best_threshold_from_results(
        y_true, [result]
    )
    return pipeline(
        result,
        [
            compute_classification_metrics_from_result(y_true, threshold=chosen_threshold),
            compute_ci_for_metrics_collection,
        ],
    )


def get_roc_point_by_threshold(
        threshold: float,
        fpr: np.array,
        tpr: np.array,
        thresholds: np.array,
) -> Tuple[float, float]:
    first_index = find_index(lambda _index: _index >= threshold, thresholds, reverse=True)
    second_index = first_index if first_index == len(thresholds) - 1 else first_index + 1

    first_threshold = thresholds[first_index]
    second_threshold = thresholds[second_index]
    ratio = (threshold - second_threshold) / (first_threshold - second_threshold) if (
            second_threshold != first_threshold
    ) else 1
    return (
        ((fpr[second_index] * (1 - ratio)) + (fpr[first_index] * ratio)),
        (tpr[second_index] * (1 - ratio) + tpr[first_index] * ratio),
    )


def compute_threshold_averaged_roc(
        y_true: Series, results: List[ModelCVResult]
) -> Tuple[np.array, np.array, np.array]:
    def roc_curve_for_fold(y_score):
        _fpr, _tpr, thresholds = roc_curve(y_true.loc[y_score.index], get_1_class_y_score(y_score))
        return _fpr, _tpr, thresholds

    roc_curves = list(
        flatten(
            [[roc_curve_for_fold(y_score) for y_score in result['y_scores']] for result in results]
        )
    )

    all_thresholds = sorted(list(flatten([roc[2] for roc in roc_curves])), reverse=True)

    def get_merged_roc_point(
            _roc_curves: List[Tuple[np.array, np.array, np.array]], threshold: float
    ) -> Tuple[float, float]:
        if threshold > 1:
            threshold = 1

        merged_fpr, merged_tpr = pipe(
            _roc_curves,
            map(lambda curve: get_roc_point_by_threshold(threshold, *curve)),
            list,
            partial(np.mean, axis=0),
        )

        return merged_fpr, merged_tpr

    merged_curve = [get_merged_roc_point(roc_curves, threshold) for threshold in all_thresholds]
    fpr, tpr = list(unzip(merged_curve))

    indexes_to_delete = []
    for index, _ in enumerate(all_thresholds):
        try:
            if fpr[index] == fpr[index + 1] or fpr[index + 1] < fpr[index]:
                indexes_to_delete.append(index)
        except IndexError:
            pass

    def remove_indexes(iterable: Iterable, indexes: List[int]) -> Iterable:
        return pipe(
            iterable,
            add_index,
            filter(decorate_unpack(lambda i, _: i not in indexes)),
            map(get(1)),
            list,
        )

    return (
        np.array(remove_indexes(fpr, indexes_to_delete)),
        np.array(remove_indexes(tpr, indexes_to_delete)),
        np.array(remove_indexes(all_thresholds, indexes_to_delete)),
    )


def get_1_class_y_score(y_score: DataFrame) -> Series:
    return y_score.iloc[:, 1]


class ConfusionMetrics(DataStructure):
    recall: float
    precision: float
    f1: float
    fpr: float
    tnr: float
    fnr: float
    npv: float

    def __init__(self, recall, precision, fpr, tnr, fnr, npv):
        self.fnr = fnr
        self.tnr = tnr
        self.recall = recall
        self.precision = precision
        self.fpr = fpr
        self.npv = npv
        try:
            self.f1 = 2 / ((1 / precision) + (1 / recall))
        except ZeroDivisionError:
            self.f1 = 0


def get_metrics_from_confusion_matrix(confusion_matrix) -> ConfusionMetrics:
    try:
        npv = confusion_matrix.tn / (confusion_matrix.tn + confusion_matrix.fn)
    except ZeroDivisionError:
        npv = 0

    return ConfusionMetrics(
        precision=(confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fp)) if
        (confusion_matrix.tp + confusion_matrix.fp) > 0 else NaN,
        recall=(confusion_matrix.tp / (confusion_matrix.tp + confusion_matrix.fn)) if
        (confusion_matrix.tp + confusion_matrix.fn) > 0 else NaN,
        fpr=confusion_matrix.fp / (confusion_matrix.fp + confusion_matrix.tn),
        tnr=confusion_matrix.tn / (confusion_matrix.fp + confusion_matrix.tn),
        fnr=confusion_matrix.fn / (confusion_matrix.fn + confusion_matrix.tp),
        npv=npv,
    )


def get_confusion_from_threshold(
        y: Series, scores: Series, threshold: float = 0.5
) -> ConfusionMatrix:
    fn = 0
    tn = 0
    tp = 0
    fp = 0

    for index, score in scores.items():
        if score < threshold:
            if y.loc[index] == 1:
                fn += 1
            elif y.loc[index] == 0:
                tn += 1
        elif score >= threshold:
            if y.loc[index] == 1:
                tp += 1
            elif y.loc[index] == 0:
                fp += 1

    matrix = ConfusionMatrix(
        fn=fn,
        tn=tn,
        tp=tp,
        fp=fp,
    )

    return matrix


@dataclass
class PRPoint(DataStructure):
    threshold: float
    metrics: Optional[ConfusionMetrics]


@dataclass
class ROCPoint(DataStructure):
    tpr: float
    tnr: float
    threshold: Optional[float] = 0


def get_metrics_from_roc(curve, tpr_threshold=0.8) -> ROCPoint:
    interpolation_roc = interp1d(curve[1], curve[0], assume_sorted=False, kind='linear')
    return ROCPoint(
        tnr=1 - interpolation_roc(tpr_threshold).flat[0],
        tpr=tpr_threshold,
    )


def get_metrics_from_pr(curve, tpr_threshold=0.8) -> PRPoint:
    index, tpr_final = find_index_right(lambda _, tpr: tpr >= tpr_threshold, curve[1])
    return PRPoint(threshold=float64(curve[2][index]), metrics=None)


def get_end_to_end_metrics_table(
        y_true, results_for_methods_optimized, results_for_methods_default
):
    metrics_for_methods_optimized = valmap(
        lambda r: compute_classification_metrics_from_results_with_statistics(y_true, r),
        results_for_methods_optimized
    )

    metrics_for_methods_default = valmap(
        lambda r: compute_classification_metrics_from_results_with_statistics(y_true, r),
        results_for_methods_default
    )

    return metrics_for_methods_optimized, metrics_for_methods_default


def get_si_k_evaluation(
        X_all, range_n_clusters, protocol, features_for_k: List[Union[List[str], str]]
):
    is_flat_features = all((not isinstance(item, list) for item in features_for_k))

    def get_si_point(_k: int) -> Optional[float]:
        _METRICS = "si"
        if is_flat_features:
            _X = X_all[features_for_k]
        else:
            _features = features_for_k[_k]
            if _features is None:
                return None
            else:
                _X = X_all[_features]

        _y_pred = protocol.algorithm(_X, _k)
        _score = protocol.measure_internal_metrics(_X, _y_pred)
        return _score[_METRICS]

    points: List[Union[None, float]] = list(map(get_si_point, range_n_clusters))

    return points


def compare_and_format_results(
        y_true: Series,
        results_for_methods: Dict[str, List[ModelCVResult]],
        include: Tuple[str] = (
                'balanced_accuracy', 'roc_auc', 'recall', 'fpr', 'f1', 'average_precision'
        ),
) -> str:
    metrics_for_methods = valmap(
        lambda r: compute_classification_metrics_from_results_with_statistics(y_true, r), results_for_methods
    )

    def get_line(method: str, metrics: ClassificationMetricsWithStatistics):
        return [format_method(method), *[metrics[metric].mean for metric in include]]

    lines = sorted(
        [get_line(method, metrics) for method, metrics in metrics_for_methods.items()],
        key=get(1),
        reverse=True,
    )

    max_by_column = [
        None if index == 0 else max(pluck(index, lines)) for index in range(len(include) + 1)
    ]

    lines_with_differences = [
        list(
            flatten(
                [
                    item if item_index == 0 else [item, item - max_by_column[item_index]]
                    for item_index, item in enumerate(line)
                ]
            )
        )
        for line in lines
    ]

    return tabulate_formatted(
        format_structure(
            format_decimal,
            [
                ['', *flatten(map(lambda metric: [format_metric_short(metric), ''], include))],
                *lines_with_differences,
            ],
        )
    )


def get_list_of_scores_from_repeated_cv_results(
        repeated_cv_results: List[ModelCVResult]
) -> List[Series]:
    return list(flatten([repeat['y_scores'] for repeat in repeated_cv_results]))


def average_list_of_confusion_matrices(
        matrices: List[ConfusionMatrix]
) -> ConfusionMatrixWithStatistics:
    return pipe(
        matrices,
        partial(map, object2dict),
        list,
        average_list_dicts,
        partial(valmap, lambda value: ValueWithStatistics(mean=value[0], std=value[1], ci=None)),
        lambda matrix: ConfusionMatrixWithStatistics(**matrix),
    )


def partial_dependency_analysis(
        method: Method, X: DataFrame, y: Series, features: List = None
) -> None:
    if features is None:
        features = list(X.columns)
    _pipeline = method.get_pipeline()
    _pipeline.fit_transform(X, y)
    plot_partial_dependence(_pipeline, X, features, target=y)
    pass


def ale_analysis(model: Estimator, X: DataFrame, features: List = None, **ale_args) -> None:
    if features is None:
        features = list(X.columns)

    for feature in features:
        print(feature)
        ale(
            X=X,
            model=model,
            feature=[feature],
            include_CI=True,
            C=0.95,
            **ale_args,
        )
        pyplot.show()


def visualise_trees(models, columns):
    columns = [list(range(
        1,
        len(models[0][-1].estimators_) + 1,
    ))] + list(
        map(
            lambda model: list(
                map(
                    lambda estimator:
                    # str(weight__estimator[0]) + "\n" +
                    export_text(estimator, show_weights=True, feature_names=list(columns)) + '\n',
                    model[-1].estimators_
                )
            ),
            models,
        )
    )
    transposed = list(map(list, moves.zip_longest(*columns, fillvalue='-')))
    return tabulate.tabulate(transposed, [f'fold {i}' for i in range(1, 11)])


def get_roc_from_fold(y, result):
    y_score = result['y_predict_probabilities'].iloc[:, 1]
    return roc_curve(y.loc[y_score.index], y_score)


def get_pr_from_fold(y, result):
    y_score = result['y_predict_probabilities'].iloc[:, 1]
    return precision_recall_curve(y.loc[y_score.index], y_score)
