import asyncio
import hashlib
import warnings
from random import shuffle
from typing import List, Any, Union, Tuple, Callable, Iterable

from frozendict import frozendict
from numpy.core.multiarray import ndarray
from pandas import DataFrame, Series
from pyramda import pick
from sklearn.model_selection import KFold, GridSearchCV
from toolz import compose
from toolz.curried import map, get

from api.api_client import api_send, external_event, event, output_ready, training_curve_point_ready, \
    TrainingCurvePoint, optimization_configuration_ready
from api.api_functions import output_feature_importance, output_curves_folds, output_feature_importance_table
from api.api_utils import json_serialize_types, json_deserialize_types
from arguments import get_params
from cache import memory_l2
from methods.parameter_space import configurations_to_grid_search, configuration_to_params, instantiate_configuration, \
    generate_configuration_series, get_defaults
from utils import warning, log, load_input_data, load_global_config, extract_features_and_label, ignore_futures, Timer, \
    object2dict
from custom_types import ClassificationMetrics
from evaluation_functions import ModelCVResult, cross_validate_model, ModelResultCurves, \
    compute_curves, \
    compute_classification_metrics_folds, cross_validate_model_cached, join_folds_cv_result, compute_classification_metrics, \
    get_1_class_y_score
from methods.functions import lvddf_to_1_class, lvddf_to_1_class_proba


def initialize() -> None:
    ignore_futures()
    warnings.simplefilter(action='ignore', category=FutureWarning)


async def optimize_generic(
    get_model,
    get_parameter_space,
    output,
    model_type,
    features,
    label,
    parallel=True,
    fast=None,
):
    fast = fast if fast is not None else get_params('fast')

    all_configurations = list(generate_configuration_series(get_parameter_space()))
    log(f'Number of configurations: {len(all_configurations)}')
    shuffle(all_configurations)
    for configuration in all_configurations:
        await random_search_worker(
            get_model, model_type, features, label, configuration, output, parallel, fast
        )


def append_data(identifier, param):
    raise Exception('Not implemented')


async def random_search_worker(
    get_model, model_type, features, label, configuration, output, parallel, fast
):
    log("Random search starting", level=1)
    # noinspection PyBroadException
    try:
        timer = Timer()
        X, y, classifier = make_model(
            get_model, features, label, configuration['pipeline'], configuration['reduce_classes']
        )

        results: ModelCVResult = cross_validate_model_cached(
            X, y, classifier, parallel=parallel, fast=fast, n_jobs=5
        )

        result_for_comparison, y_for_comparison = get_result_for_comparison(
            configuration['reduce_classes'], results, y, label
        )

        tab_data_output, records_output = output_all(result_for_comparison, y_for_comparison)

        event_data = optimization_configuration_ready(
            results=tab_data_output,
            records=records_output,
            parameters=configuration,
            features=features,
            model_type=model_type,
            label=label,
            elapsed=timer.elapsed_cpu()
        )

        identifier = (
            hash_features(features),
            label,
            model_type,
        )

        append_data(identifier, event_data['payload'])
        await output(event_data)
        log("Optimization iteration finished`")
    except Exception as e:
        warning(e)
        raise e


async def run_training_curve_generic(
    get_model,
    output,
    configuration,
    features,
    label,
    tab,
    cv=10,
    parallel=True,
):
    X, y, classifier = make_model(
        get_model, features, label, configuration['pipeline'], configuration['reduce_classes']
    )
    n = len(X)
    splits_count = 20
    splits = [round(n / splits_count) * i for i in range(1, splits_count + 1)]
    for i, split_size in enumerate(splits):
        result = cross_validate_model(
            X.iloc[:split_size], y.iloc[:split_size], classifier, cv=cv, parallel=parallel
        )
        result_for_comparison, y_for_comparison = get_result_for_comparison(
            configuration['reduce_classes'],
            result,
            y,
            label,
        )
        metrics_test = compute_classification_metrics_folds(result_for_comparison.y_scores, y_for_comparison)
        metrics_train = compute_classification_metrics_folds(
            result_for_comparison.y_train_scores, y_for_comparison
        )
        await output(
            training_curve_point_ready(
                TrainingCurvePoint(
                    n_samples=split_size, metrics_test=metrics_test, metrics_train=metrics_train
                ),
                tab,
                finished=len(splits) - 1 == i
            )
        )


async def run_repeated_nested_cross_validation_generic(
    get_parameter_space: Callable,
    features: List[str],
    label: str,
):
    parameter_space = get_parameter_space()
    X, y = get_x_y(features, label, reduce_classes=True)
    p_grid = configurations_to_grid_search(
        map(
            compose(configuration_to_params, instantiate_configuration, get('pipeline')),
            list(generate_configuration_series(parameter_space))[:3],
        ),
    )
    inner_cv = KFold(n_splits=2, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=2, shuffle=True, random_state=1)
    clf = GridSearchCV(estimator=get_model(), param_grid=p_grid, cv=inner_cv, n_jobs=-1, verbose=2)
    clf.fit(X, y)
    non_nested_scores = clf.best_score_


def output_all(result_for_comparison, y_for_comparison) -> Tuple[dict, List[dict]]:
    payload = output_results_full_vector(result_for_comparison, y_for_comparison)
    records = [
        output_feature_importance(result_for_comparison),
        output_curves_folds(result_for_comparison, y_for_comparison),
        output_feature_importance_table(result_for_comparison)
    ]
    return payload, records


output_all_cached = memory_l2.cache(output_all)


def get_result_for_comparison(
    reduce_classes: bool,
    result: ModelCVResult,
    y: Series,
    label: str,
) -> Tuple[ModelCVResult, Series]:
    config = load_global_config()
    try:
        reduce_class_label = config['reduce_class_label']
    except KeyError:
        reduce_class_label = None

    if not reduce_classes and label == reduce_class_label:
        result_for_comparison = lvddf_to_1_class_result(result)
        y_for_comparison = lvddf_to_1_class(y)
    else:
        result_for_comparison = result
        y_for_comparison = y
    return result_for_comparison, y_for_comparison


def make_model(get_model, features, label, configuration, reduce_classes, path=None):
    X, y = get_x_y(features, label, reduce_classes=reduce_classes, path=path)
    classifier = get_model()
    set_model_params(configuration, classifier)
    return X, y, classifier


def set_model_params(params, classifier):
    classifier.set_params(
        **configuration_to_params(instantiate_configuration(json_deserialize_types(params)))
    )


def hash_features(features):
    return str(len(features)) + "_" + str(
        int(hashlib.sha256("_".join(sorted(features)).encode('utf-8')).hexdigest(), 16) % 10**8
    )


def get_x_y(features, label, data=None, reduce_classes=True, path=None):
    config = load_global_config()
    if data is None:
        data = load_input_data(path)

    try:
        combined_label, combined_from = config['combined_label'], config['combined_from']
    except KeyError:
        X, y = extract_features_and_label(
            data,
            label,
            features,
        )
    else:
        if label == combined_label:
            X, _ = extract_features_and_label(
                data,
                features=features,
            )
            class1 = data[combined_from[0]]
            class2 = data[combined_from[1]]
            y = lvddf_to_1_class(class1).combine(class2, lambda val1, val2: val1 or val2)
        else:
            X, y = extract_features_and_label(
                data,
                label,
                features,
            )

    try:
        reduce_class_label = config['reduce_class_label']
    except KeyError:
        pass
    else:
        if label == reduce_class_label and reduce_classes:
            y = lvddf_to_1_class(y)

    return X, y


def filter_X_y(callback: Callable, X: DataFrame, y: Series) -> Tuple[DataFrame, Series]:
    filtered_mask = callback(X)
    return X[filtered_mask], y[filtered_mask]


def get_disjoint_datasets(
    callback: Callable,
    X: DataFrame,
    y: Series,
) -> Tuple[DataFrame, Series, DataFrame, Series]:
    filtered_mask = callback(X)
    return X[~filtered_mask], y[~filtered_mask], X[filtered_mask], y[filtered_mask]


def get_data(*args, **kwargs):
    X, y = get_x_y(*args, **kwargs)
    data = X.copy()
    data['y'] = y
    return data


async def run_cli_parameter_space(
    model_type,
    get_parameter_space,
):
    await api_send(
        external_event(
            event(
                "PARAMETER_SPACE_READY", {
                    'type': model_type,
                    'space': json_serialize_types(get_parameter_space()),
                }
            )
        )
    )


def run_model_from_cli(model_type, run, get_parameter_space, configuration=None, **kwargs):
    asyncio.get_event_loop().run_until_complete(
        run(
            **{
                **pick(['features', 'label'], load_global_config()),
                **{
                    'output': lambda message: api_send(external_event(output_ready(message))),
                    'configuration': configuration or get_defaults(get_parameter_space()),
                    'tab': model_type,
                    **kwargs,
                },
            },
        )
    )


def output_results_full_vector(result: ModelCVResult, y_true: Series) -> dict:
    y_predict, y_score = get_full_vector_result_vectors(result)
    curves: ModelResultCurves = compute_curves(y_score, y_true)
    return {
        'metrics': object2dict(output_metrics(result, y_true)),
        'roc': {
            'fpr': curves.curve_horizontal,
            'tpr': curves.curve_vertical_roc,
        },
        'recall_precision': {
            'recall': curves.curve_horizontal,
            'precision': curves.curve_vertical_recall_precision,
        }
    }


def output_metrics(result: ModelCVResult, y_true: Series) -> ClassificationMetrics:
    y_predict, y_score = get_full_vector_result_vectors(result)
    return compute_classification_metrics(y_score, y_true)


def get_full_vector_result_vectors(result) -> Tuple[Series, Series]:
    result_full_vector = join_folds_cv_result(result)
    y_score = get_1_class_y_score(result_full_vector['y_test_score'])
    return result_full_vector['y_test_predict'], y_score


def get_subset_of_second_by_index_of_first(series1: Series, series2: Series) -> Series:
    return series2.loc[series1.index]


def stop_loading(tab, tab_data):
    return {
        'tab': {
            'key': tab,
            'loading': False,
            'payload': tab_data,
            'isModel': True,
        }
    }


def start_loading(tab):
    return ({
        'tab': {
            'key': tab,
            'loading': True,
        }
    })


def get_categorical_features(data: Union[DataFrame, ndarray], number: int = 5) -> List[str]:
    data_frame = DataFrame(data)
    columns: List[Any] = []
    for column in data_frame.columns:
        if data_frame[column].nunique() <= number:
            columns.append(column)
    return columns


def convert_columns_to_string(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    new_df = df.copy()
    for column in columns:
        new_df[column] = df[column].astype(str)
    return new_df


def convert_categorical_to_strings(df: DataFrame) -> Tuple[DataFrame, List[str]]:
    categorical_features = get_categorical_features(df)
    return convert_columns_to_string(df, categorical_features), categorical_features


def lvddf_to_1_class_result(result: ModelCVResult):
    return ModelCVResult(
        y_predicts=map(lvddf_to_1_class, result['y_predicts']),
        y_train_predicts=map(lvddf_to_1_class, result['y_train_predicts']),
        y_scores=map(lvddf_to_1_class_proba, result['y_scores']),
        y_train_scores=map(lvddf_to_1_class_proba, result['y_train_scores']),
        feature_importance=result['feature_importance'],
        models=result['models'],
        elapsed=result['elapsed'],
    )


def frozendict_r(value: Union[dict, Any]) -> frozendict:
    if isinstance(value, dict):
        return frozendict(
            {k: frozendict(v) if isinstance(v, dict) else v
             for k, v in value.items()}
        )
    else:
        return value
