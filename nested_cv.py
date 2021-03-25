import dataclasses
import logging
import multiprocessing
import random
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import partial
from multiprocessing import Process
from typing import Any, List, Callable, Dict, TypedDict, Optional, Tuple, Iterable, Mapping

from functional_pipeline import pipeline
from hyperopt import fmin, tpe, STATUS_OK, Trials, atpe
from imblearn.over_sampling import RandomOverSampler
from numpy import mean
from pandas import DataFrame, Series
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from termcolor import colored
from toolz import pluck, identity
from toolz.curried import map, get_in

from api.api_utils import json_deserialize_types
from cache import save_data, load_data
from utils import log, load_global_config, Timer, Counter, hash_dict, list_of_dicts_to_dict_of_lists, dict_mean, \
    call_and_push_to_queue, empty_dict
from custom_types import SupervisedPayload, ClassificationMetricsWithSTD, Estimator
from evaluation_functions import result_from_fold_results, cross_validate_model_sets_args, \
    compute_classification_metrics_folds, cross_validate_model_fold_args, ModelCVResult, get_classification_metrics, \
    ModelResult, cross_validate_model_sets
from functional import mapl
from methods.methods_utils import get_x_y, output_all, output_metrics
from methods.parameter_space import generate_configuration_series, dict_configuration_to_params, get_defaults, \
    configuration_to_params

evaluation_metric = 'average_precision'
displayed_metrics = ['roc_auc', 'average_precision']
random_state = 454


def get_protocols():
    config = load_global_config()

    return {
        'evaluation_inner_10': {
            'features': config['limited_features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 10,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'evaluation_2x5': {
            'features': config['limited_features'],
            'random_state': 454,
            'outer_cv': 2,
            'outer_repeats': 10,
            'inner_cv': 2,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'evaluation_inner_3': {
            'features': config['limited_features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 3,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'default': {
            'features': config['limited_features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 3,
            'inner_repeats': 0,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'default_rf_top_features': {
            'features': [
                'TRGL', 'SOCK', 'SK', 'K', 'SCRT', 'LFERR', 'LPRA', 'BSUG', 'QTCD', 'BW', 'LCRTSL',
                'SKINF', 'WHR', 'MBP', 'WAISTC', 'DBP', 'RA1_V5', 'LLEPT', 'TA_AVG', 'HHT',
                'RA1_AVL', 'BMI', 'SBP', 'PP', 'AGE'
            ],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 3,
            'inner_repeats': 0,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'default_xgboost_top_features': {
            'features': [
                'HCHOL', 'BW', 'LGGT', 'MBP', 'TRT_CEB', 'SNA', 'LCRTSL', 'RBC', 'K', 'QTCD', 'SUA',
                'SEX', 'RA1_V5', 'HGB', 'RA1_AVL', 'SBP', 'TRT_DD', 'TA_AVG', 'LLEPT', 'PP',
                'WAISTC', 'BMI', 'SOCK', 'HHT', 'AGE'
            ],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 3,
            'inner_repeats': 0,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'inner_10_xgboost_top_features': {
            'features': [
                'HCHOL', 'BW', 'LGGT', 'MBP', 'TRT_CEB', 'SNA', 'LCRTSL', 'RBC', 'K', 'QTCD', 'SUA',
                'SEX', 'RA1_V5', 'HGB', 'RA1_AVL', 'SBP', 'TRT_DD', 'TA_AVG', 'LLEPT', 'PP',
                'WAISTC', 'BMI', 'SOCK', 'HHT', 'AGE'
            ],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 10,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'inner_10_preselected': {
            'features': config['preselected_features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 10,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'default_preselected': {
            'features': config['preselected_features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 10,
            'inner_repeats': 0,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'evaluation_full': {
            'features': config['features'],
            'random_state': 454,
            'outer_cv': 10,
            'outer_repeats': 10,
            'inner_cv': 3,
            'inner_repeats': 1,
            'reduce_configurations': None,
            'evaluation_metric': evaluation_metric,
        },
        'quick': {
            'features': config['limited_features'],
            'random_state': 454,
            'outer_cv': 2,
            'outer_repeats': 2,
            'inner_cv': 2,
            'inner_repeats': 2,
            'reduce_configurations': 1,
            'evaluation_metric': evaluation_metric,
        },
    }


def evaluate_repeated_nested_cross_validation(
    get_pipeline,
    get_parameter_space,
    payload: SupervisedPayload,
) -> List[ModelCVResult]:
    outer_cv = 10
    inner_cv = 10
    outer_repeats = 1
    inner_repeats = 1
    reduce_configurations: Optional[int] = None
    parallel = True
    counter = Counter()
    X, y = get_x_y(
        payload['features'],
        payload['label'],
        reduce_classes=True,
    )

    random.seed(random_state)

    all_configurations = list(generate_configuration_series(get_parameter_space()))

    configurations = random.sample(
        all_configurations, reduce_configurations
    ) if reduce_configurations else all_configurations
    #
    dimensionality_cv = outer_repeats * outer_cv * inner_cv * inner_repeats
    dimensionality_configurations = len(configurations)
    dimensionality_total = dimensionality_cv * dimensionality_configurations
    #
    # log(
    #     f'Configurations count: {dimensionality_configurations}, Cross validation dim.: {dimensionality_cv}, '
    #     f'Total Dimensionality: {dimensionality_total} '
    # )
    #
    # if reduce_configurations:
    #     log(f'Reduced configurations: {percents(reduce_configurations, len(all_configurations))}')

    timer = Timer()

    # try:
    #     with open(get_data_path(identifier), 'rb') as f:
    #         intermediate_data = pickle.load(f)
    #     if isinstance(intermediate_data, list):
    #         intermediate_data = {
    #             'outer_k_fold_folds': [],
    #             'outer_repeated_results': intermediate_data,
    #         }
    #
    #     outer_repeated_results = intermediate_data['outer_repeated_results']
    #     counter.count = int(
    #         ((dimensionality_total / outer_repeats) * len(outer_repeated_results)) + (
    #             inner_cv * inner_repeats * dimensionality_configurations *
    #             len(intermediate_data['outer_k_fold_folds'])
    #         )
    #     )
    #     counter.initial = counter.count
    #     resumed = True
    #     log(f'Resuming on iteration {str(len(outer_repeated_results)+1)}')
    # except FileNotFoundError:
    #     resumed = False
    outer_repeated_results: List[Any] = []

    # Outer repeated
    for outer_repeats_random_state in range(len(outer_repeated_results), outer_repeats):
        logging.debug('Starting outer repeat')
        outer_k_fold = get_outer_kfold(outer_cv, outer_repeats_random_state)

        # if resumed:
        #     resumed = False
        #     outer_k_fold_folds = intermediate_data['outer_k_fold_folds']
        # else:
        outer_k_fold_folds: List[Any] = []

        outer_k_fold_splits = list(outer_k_fold.split(X, y))
        # Outer k-fold
        for train_index_outer, test_index_outer in outer_k_fold_splits[len(outer_k_fold_folds):]:

            logging.debug('Starting outer k fold')
            # Inner repeated
            inner_repeated_results = []
            if inner_repeats != 0:
                for inner_repeat_random_state in range(inner_repeats):
                    logging.debug('Starting inner repeat')
                    # Configurations
                    inner_configurations = {}
                    for configuration in configurations:
                        classifier = get_pipeline()
                        parameters = dict_configuration_to_params(configuration)
                        classifier.set_params(**parameters)

                        # Inner k-fold
                        # noinspection Mypy
                        inner_configurations[hash_dict(configuration)] = nested_cv(
                            payload['features'], payload['label'], classifier,
                            inner_repeat_random_state, train_index_outer, inner_cv,
                            dimensionality_total, timer, counter, parallel
                        )

                    inner_repeated_results.append(inner_configurations)

                configurations_metrics, best_inner_configuration = get_best_inner_configuration(
                    evaluation_metric, y, inner_repeated_results
                )
            else:
                best_inner_configuration = {'configuration': get_defaults(get_parameter_space())}
                del (best_inner_configuration['configuration']['pipeline']['classifier'])
                configurations_metrics = None

            try:
                logging.debug(
                    'Best configuration (average precision): ' +
                    str(round(best_inner_configuration['metrics']['average_precision'], 3))
                )
                logging.debug(
                    'Best configuration (roc auc): ' +
                    str(round(best_inner_configuration['metrics']['roc_auc'], 3))
                )
            except KeyError:
                pass
            #
            classifier = get_pipeline()
            classifier.set_params(
                **dict_configuration_to_params(best_inner_configuration['configuration'])
            )
            # outer_fold = cross_validate_model_fold_args(
            #     classifier,
            #     get_x_y=partial(
            #         get_x_y,
            #         features=features,
            #         label=label,
            #         reduce_classes=True,
            #     ),
            #     train_index=train_index_outer,
            #     test_index=test_index_outer,
            # )

            # Xgboost fais, when not run in separate process
            queue: Any = multiprocessing.Queue()
            p = Process(
                target=call_and_push_to_queue,
                kwargs=dict(
                    partial_func=partial(
                        cross_validate_model_fold_args,
                        classifier=classifier,
                        get_x_y=partial(
                            get_x_y,
                            features=payload['features'],
                            label=payload['label'],
                            reduce_classes=True,
                        ),
                        train_index=train_index_outer,
                        test_index=test_index_outer,
                        return_model=False,
                    ),
                    queue=queue,
                )
            )
            p.start()
            p.join()

            outer_fold = queue.get()

            outer_k_fold_folds.append(
                {
                    'outer_fold': outer_fold,
                    'configurations_metrics': configurations_metrics
                }
            )
            print('done')
            # with open(get_data_path(identifier), 'wb') as f:
            #     pickle.dump(
            #         {
            #             'outer_repeated_results': outer_repeated_results,
            #             'outer_k_fold_folds': outer_k_fold_folds,
            #         }, f
            #     )

        outer_k_fold_results = result_from_fold_results(pluck('outer_fold', outer_k_fold_folds))
        output_payload, records = output_all(outer_k_fold_results, y)
        outer_repeated_results.append(
            {
                'result': outer_k_fold_results,
                'folds': outer_k_fold_folds,
                'result_output': {
                    'payload': output_payload,
                    'records': records,
                },
            }
        )
        # with open(get_data_path(identifier), 'wb') as f:
        #     pickle.dump(
        #         {
        #             'outer_repeated_results': outer_repeated_results,
        #             'outer_k_fold_folds': [],
        #         }, f
        #     )
        # log('Saving data')

    return [repeat['result'] for repeat in outer_repeated_results]


Configuration = Dict[str, Any]


class ValueWithSTD(TypedDict):
    mean: float
    std: Optional[float]


class ObjectiveFunctionResult(TypedDict):
    configuration: Dict
    metrics: Optional[ClassificationMetricsWithSTD]
    result: Optional[ModelCVResult]
    payload: Optional[Any]


class ObjectiveFunctionResultWithPayload(TypedDict):
    chosen: ObjectiveFunctionResult
    payload: Any


ObjectiveFunction = Callable[[Configuration], ObjectiveFunctionResult]


class OutputConfiguration(TypedDict):
    configuration: Configuration
    metrics: Optional[ClassificationMetricsWithSTD]


class ReturnHyperParameters(ABC):

    @property
    @abstractmethod
    def iterations(self):
        ...

    @abstractmethod
    def get(self, objective: ObjectiveFunction) -> ObjectiveFunctionResultWithPayload:
        ...


class BayesianOptimization(ReturnHyperParameters):

    def __init__(
        self,
        space: Any,
        iterations: int = 50,
        target_metric='roc_auc',
        return_trials: bool = False
    ):
        self.space = space
        self.target_metric = target_metric
        self._iterations = iterations
        self.return_trials = return_trials

    @property
    def iterations(self) -> int:
        return self._iterations

    def get(self, objective: ObjectiveFunction) -> ObjectiveFunctionResultWithPayload:

        def objective_wrapped(configuration: Configuration) -> Dict:
            objective_result: ObjectiveFunctionResult = objective(configuration)
            return {
                'loss': 1 - objective_result['metrics'][self.target_metric][0],
                'loss_variance': objective_result['metrics'][self.target_metric][1],
                'payload': objective_result,
                'status': STATUS_OK,
            }

        trials = Trials()

        fmin(
            objective_wrapped,
            space=self.space,
            trials=trials,
            algo=atpe.suggest,
            verbose=True,
            max_evals=self.iterations,
            show_progressbar=True,
        )
        best_item = sorted(
            trials.results,
            key=lambda item: item['loss'],
        )[0]

        return ObjectiveFunctionResultWithPayload(
            chosen=best_item['payload'], payload=trials if self.return_trials else None
        )


class StaticHyperParameters(ReturnHyperParameters):

    def __init__(self, configuration: Dict):
        self.configuration = configuration

    @property
    def iterations(self):
        return 1

    def get(self, objective: ObjectiveFunction) -> ObjectiveFunctionResultWithPayload:
        objective_result = objective(self.configuration)

        return ObjectiveFunctionResultWithPayload(
            chosen=ObjectiveFunctionResult(
                configuration=self.configuration,
                metrics=objective_result['metrics'],
                result=objective_result['result'],
                payload=None
            ),
            payload=None
        )


class DefaultHyperParameters(StaticHyperParameters):

    def __init__(self):
        super().__init__({})


class NestedEvaluationProtocol(TypedDict):
    outer_repeats: int
    outer_cv: int
    inner_repeats: int
    inner_cv: int


def is_protocol_nested(protocol: NestedEvaluationProtocol) -> bool:
    return protocol['inner_repeats'] != 0


def evaluate_nested_cross_validation(
    get_pipeline,
    optimize: ReturnHyperParameters,
    payload: SupervisedPayload,
    protocol=None,
    parallel: bool = True,
    cache_key: Optional[str] = None,
) -> List[ModelCVResult]:
    if protocol is None:
        protocol = NestedEvaluationProtocol(
            outer_repeats=1,
            outer_cv=10,
            inner_repeats=1,
            inner_cv=10,
        )

    def save_state():
        if cache_path:
            logging.info('Saving...')
            save_data(cache_path, {'outer_folds': outer_folds, 'outer_repeats': outer_repeats})

    def get_state():
        try:
            if cache_path:
                logging.info("Loading state...")
            data = load_data(cache_path)
        except FileNotFoundError:
            return [], []
        else:
            return data['outer_folds'], data['outer_repeats']

    cache_path: Optional[List]

    if cache_key:
        cache_path = [
            'evaluate_nested_cross_validation',
            hash_dict(protocol) + hash_dict(payload),
            cache_key,
        ]
    else:
        cache_path = None

    outer_repeats: List[ModelCVResult]
    outer_folds: List[ModelResult]

    outer_folds, outer_repeats = get_state()

    X, y = get_x_y(
        payload['features'],
        payload['label'],
        reduce_classes=True,
    )

    random.seed(random_state)
    dimensionality_cv = protocol['outer_repeats'] * protocol['outer_cv'] * (
        1 if not is_protocol_nested(protocol) else protocol['inner_cv'] * protocol['inner_repeats']
    )
    dimensionality_total = dimensionality_cv * optimize.iterations
    counter = Counter(
        int(
            ((dimensionality_total / protocol['outer_repeats']) * len(outer_repeats)) + (
                protocol['inner_cv'] * protocol['inner_repeats'] * optimize.iterations *
                len(outer_folds)
            )
        )
    )
    timer = Timer()

    # Outer repeated
    for outer_repeats_random_state in range(len(outer_repeats), protocol['outer_repeats']):
        logging.debug('Starting outer repeat')
        outer_k_fold = get_outer_kfold(protocol['outer_cv'], outer_repeats_random_state)

        outer_k_fold_splits = list(outer_k_fold.split(X, y))

        # Outer k-fold
        for train_index_outer, test_index_outer in outer_k_fold_splits[len(outer_folds):]:
            logging.debug('Starting outer k fold')

            inner_repeated_results: List[ObjectiveFunctionResultWithPayload] = []

            # Inner repeats
            for inner_repeat_random_state in range(protocol['inner_repeats']):
                logging.debug('Starting inner repeat')

                def evaluate_configuration(configuration: Configuration) -> ObjectiveFunctionResult:
                    pipeline_to_evaluate = get_pipeline()
                    pipeline_to_evaluate.set_params(
                        **configuration_to_params(json_deserialize_types(configuration))
                    )
                    cv_result = nested_cv(
                        payload['features'], payload['label'], pipeline_to_evaluate,
                        inner_repeat_random_state, train_index_outer, protocol['inner_cv'],
                        dimensionality_total, timer, counter, parallel
                    )
                    metrics = get_classification_metrics(y, cv_result)

                    return ObjectiveFunctionResult(
                        metrics=metrics,
                        configuration=configuration,
                        result=cv_result,
                        payload=None,
                    )

                inner_repeated_results.append(optimize.get(evaluate_configuration))

            optimized = inner_repeated_results[0]['chosen']

            try:
                logging.debug(
                    'Best configuration (average precision): ' +
                    str(round(optimized['metrics']['average_precision'][0], 3))
                )
                logging.debug(
                    'Best configuration (roc auc): ' +
                    str(round(optimized['metrics']['roc_auc'][0], 3))
                )
            except (KeyError, TypeError):
                pass

            optimized = inner_repeated_results[0]

            classifier = get_pipeline()
            classifier.set_params(
                **configuration_to_params(
                    json_deserialize_types(optimized['chosen']['configuration'])
                )
            )
            outer_fold = cross_validate_model_fold_args(
                classifier=classifier,
                get_x_y=partial(
                    get_x_y,
                    features=payload['features'],
                    label=payload['label'],
                    reduce_classes=True,
                ),
                train_index=train_index_outer,
                test_index=test_index_outer,
                return_model=False,
            )
            outer_folds.append(outer_fold)
            save_state()
        result: ModelCVResult = result_from_fold_results(outer_folds)
        outer_repeats.append(result)
        outer_folds = []
        save_state()
    return outer_repeats


class SimpleEvaluationProtocol(TypedDict):
    repeats: int
    cv: int
    stratified: Optional[bool]


def evaluate_method_on_sets(
    get_pipeline,
    X: DataFrame,
    y: Series,
    optimize: ReturnHyperParameters,
    splits: Iterable[Tuple[List[int], List[int]]],
    parallel: bool = True,
    filter_X_test: Callable[[DataFrame], DataFrame] = identity,
    feature_names: Optional[List[str]] = None,
    get_metrics: Callable[[Series, ModelCVResult], Any] = get_classification_metrics,
    n_jobs: int = 12,
    fit_kwargs: Mapping = empty_dict,
) -> ObjectiveFunctionResultWithPayload:

    def evaluate_configuration(configuration: Configuration) -> ObjectiveFunctionResult:
        classifier = get_pipeline()
        classifier.set_params(**configuration_to_params(json_deserialize_types(configuration)))
        result = cross_validate_model_sets(
            classifier=classifier,
            X=X,
            y=y,
            sets=splits,
            return_model=False,
            parallel=parallel,
            filter_X_test=filter_X_test,
            feature_names=feature_names,
            n_jobs=n_jobs,
            fit_kwargs=fit_kwargs,
        )
        metrics = get_metrics(y, result)

        return ObjectiveFunctionResult(
            metrics=metrics,
            configuration=configuration,
            result=result,
            payload=None,
        )

    returned_configuration: ObjectiveFunctionResultWithPayload = optimize.get(
        evaluate_configuration
    )

    return returned_configuration


def get_cv_results_from_simple_cv_evaluation(
    simple_cv_result: List[ObjectiveFunctionResultWithPayload]
) -> List[ModelCVResult]:
    return mapl(lambda item: item['chosen']['result'], simple_cv_result)


def get_outer_kfold(outer_cv, outer_repeat, stratified: bool):
    stratified_function = StratifiedKFold if stratified else KFold
    outer_k_fold = stratified_function(n_splits=outer_cv, shuffle=True, random_state=outer_repeat)
    return outer_k_fold


def get_data_path(identifier):
    return 'data/repeated_nested_cv__' + identifier


def get_outer_folds_data_path(identifier):
    return 'data/folds/repeated_nested_cv__' + identifier


def get_metric_statistic(metric, outer_repeated_results):
    get_metric = get_in(['result_output', 'payload', 'metrics', metric])
    min_score = min(map(get_metric, outer_repeated_results))
    mean_score = mean(list(map(get_metric, outer_repeated_results)))
    max_score = max(map(get_metric, outer_repeated_results))
    return max_score, mean_score, min_score


def nested_cv(
    features,
    label,
    classifier,
    inner_kfold_random_state,
    dataset_index,
    inner_cv,
    dimensionality_total=None,
    timer=None,
    counter=None,
    parallel=True
):
    nested_k_fold = KFold(n_splits=inner_cv, shuffle=True, random_state=inner_kfold_random_state)
    sets_index = list(nested_k_fold.split(dataset_index))
    sets = [
        (
            [dataset_index[index] for index in fold[0]],
            [dataset_index[index] for index in fold[1]],
        ) for fold in sets_index
    ]
    cv_timer = Timer()
    results = cross_validate_model_sets_args(
        classifier=classifier,
        sets=sets,
        get_x_y=partial(
            get_x_y,
            features=features,
            label=label,
            reduce_classes=True,
        ),
        parallel=parallel,
        return_model=False,
    )

    log(f'CV finished in {cv_timer.elapsed_cpu():.2f}s')

    if counter:
        counter.increase(inner_cv)

    if timer and dimensionality_total:
        info_elapsed(counter, dimensionality_total, timer)

    return results


def get_best_inner_configuration(metric, y, configurations_by_repeat):
    repeats_by_configuration = list_of_dicts_to_dict_of_lists(configurations_by_repeat)
    repeats_by_configuration_with_metrics = list(
        {
            configuration_hash: [
                {
                    'configuration': results['payload']['configuration'],
                    'metrics': output_metrics(results['results'], y),
                    'metrics_folds': compute_classification_metrics_folds(
                        results['results']['y_scores'], y
                    ),
                } for results in configuration_results
            ]
            for configuration_hash, configuration_results in repeats_by_configuration.items()
        }.values()
    )
    configurations_averaged = pipeline(
        repeats_by_configuration_with_metrics,
        [
            map(
                lambda repeats: {
                    'configuration': repeats[0]['configuration'],
                    'metrics': dict_mean(
                        map(lambda repeat: dataclasses.asdict(repeat['metrics']), repeats)
                    )
                }
            ),
            list,
        ],
    )
    best_configuration = sorted(
        configurations_averaged,
        key=get_in(['metrics', metric]),
        reverse=True,
    )[0]

    return repeats_by_configuration_with_metrics, best_configuration


def optimize_and_train_cv(
    get_pipeline: Callable[[], Estimator],
    X: DataFrame,
    y: Series,
    optimize: ReturnHyperParameters,
    protocol: Optional[SimpleEvaluationProtocol] = None,
    parallel: bool = True,
) -> Estimator:
    if protocol is None:
        protocol = SimpleEvaluationProtocol(cv=10, repeats=1)

    result: ObjectiveFunctionResultWithPayload = evaluate_simple_cross_validation(
        get_pipeline,
        X,
        y,
        optimize,
        protocol,
        parallel,
    )[0]

    classifier = get_pipeline().set_params(
        **configuration_to_params(result['chosen']['configuration'])
    )
    classifier.fit_transform(X, y)

    return classifier


def info_elapsed(counter, dimensionality, timer):
    counter_after_resume = counter.count - counter.initial

    def format_time(delta):
        return str(delta).split('.', 1)[0]

    print(
        colored(
            f'Processed {(counter.count / dimensionality) * 100:.2f}%: {counter.count} / {dimensionality}',
            'yellow'
        )
    )
    try:
        estimated_time = format_time(
            timedelta(
                seconds=(
                    (timer.elapsed_real() / counter_after_resume) *
                    (dimensionality - counter_after_resume)
                )
            )
        )
        log(
            f'Running: {format_time(timedelta(seconds=timer.elapsed_real()))}, '
            f'Estimated remaining time: '
            f'{estimated_time}'
        )
    except ZeroDivisionError:
        pass
