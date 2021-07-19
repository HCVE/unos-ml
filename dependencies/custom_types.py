from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Tuple, Optional, Dict, Generic, TypeVar, Mapping, Iterable, Union

from pandas import Series, DataFrame
from sklearn.pipeline import Pipeline
from typing_extensions import TypedDict

T1 = TypeVar('T1')


class Estimator(ABC):

    def fit(self, X, y) -> None:
        ...

    def predict(self, X) -> Any:
        ...

    def predict_proba(self, X) -> Any:
        ...

    def set_params(self, **kwargs):
        ...


# noinspection PyUnresolvedReferences
Estimator.register(Pipeline)


class CrossValidationType(Enum):
    NESTED = 'NESTED'
    SIMPLE = 'SIMPLE'


class SupervisedPayloadConfiguration(TypedDict):
    pipeline: Dict
    reduce_classes: bool


class SupervisedPayloadDataset(TypedDict):
    key: str
    label: str
    source_path: str
    features: List[str]
    schema: List[Dict]


class SupervisedPayload(TypedDict):
    cross_validation_type: CrossValidationType
    label: str
    optimize: bool
    features: List[str]
    configuration: SupervisedPayloadConfiguration
    dataset: SupervisedPayloadDataset
    cv_outer_repeats: int
    cv_outer_folds: int


class DictAccess:

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)


class ClassMapping(Mapping):

    def __getitem__(self, item):
        try:
            return self.__dict__[item]
        except AttributeError:
            return self[item]

    def __iter__(self):
        return (k for k in self.__dict__.keys())

    def __len__(self):
        return len(self.__dict__.keys())


class Printable:

    def __str__(self):
        return '\n'.join(
            [f'{key}: {value}' for key, value in self.__dict__.items() if not key.startswith('_')]
        )


class DataStructure(DictAccess, ClassMapping, Printable):
    ...


@dataclass(repr=False)
class ClassificationMetrics(DataStructure):
    recall: float
    precision: float
    f1: float
    tnr: float
    fpr: float
    fnr: float
    accuracy: float
    roc_auc: float
    average_precision: float
    balanced_accuracy: float
    brier_score: float
    npv: float


@dataclass(repr=False)
class RegressionMetrics(DataStructure):
    explained_variance: float
    r2: float
    mean_absolute_error: float


@dataclass(repr=False)
class RegressionMetricsWithSTD(DataStructure):
    explained_variance: Tuple[float, float]
    r2: Tuple[float, float]
    mean_absolute_error: Tuple[float, float]


@dataclass(repr=False)
class ClassificationMetricsWithSTD(DataStructure):
    recall: Tuple[float, float]
    precision: Tuple[float, float]
    f1: Tuple[float, float]
    tnr: Tuple[float, float]
    fpr: Tuple[float, float]
    fnr: Tuple[float, float]
    npv: Optional[Tuple[float, float]] = None
    accuracy: Optional[Tuple[float, float]] = None
    roc_auc: Optional[Tuple[float, float]] = None
    average_precision: Optional[Tuple[float, float]] = None
    balanced_accuracy: Optional[float] = None
    brier_score: Optional[Tuple[float, float]] = None


@dataclass
class ValueWithStatistics:
    mean: float
    std: Optional[float]
    ci: Optional[Tuple[float, float]]

    def format_to_list(self):
        from formatting import format_decimal
        return [
            self.format_mean(), f'±{format_decimal(self.std)}',
            *([self.format_ci()] if self.ci else [])
        ]

    def format_mean(self) -> str:
        from formatting import format_decimal
        return format_decimal(self.mean)

    def format_ci(self) -> str:
        from formatting import format_ci

        if not self.ci[0] or not self.ci[1]:
            return ""

        if self.ci:
            return format_ci(self.ci)
        else:
            raise AttributeError('CI not assigned')

    def format_short(self):
        if self.ci:
            return f'{self.format_mean()} ({self.format_ci()})'
        else:
            return f'{self.format_mean()}'

    def __str__(self):
        return " ".join(self.format_to_list())


@dataclass(repr=False)
class RegressionMetricsWithStatistics(DataStructure):
    explained_variance: ValueWithStatistics
    r2: ValueWithStatistics
    mean_absolute_error: ValueWithStatistics


@dataclass(repr=False)
class ClassificationMetricsWithStatistics(DataStructure):
    recall: ValueWithStatistics
    precision: ValueWithStatistics
    f1: ValueWithStatistics
    tnr: ValueWithStatistics
    fpr: ValueWithStatistics
    fnr: ValueWithStatistics
    npv: ValueWithStatistics
    accuracy: ValueWithStatistics
    roc_auc: ValueWithStatistics
    average_precision: ValueWithStatistics
    balanced_accuracy: ValueWithStatistics


class FeatureImportanceItem(TypedDict):
    feature: str
    importance: float


class ResultPayload(TypedDict):
    metrics: ClassificationMetricsWithStatistics
    feature_importance: List[FeatureImportanceItem]


@dataclass
class GenericConfusionMatrix(DataStructure, Generic[T1]):
    fn: T1
    tn: T1
    tp: T1
    fp: T1


ConfusionMatrix = GenericConfusionMatrix[float]
ConfusionMatrixWithStatistics = GenericConfusionMatrix[ValueWithStatistics]


class MethodInfo(TypedDict):
    parallel: bool
    iterations: Optional[int]


class Method:

    @staticmethod
    def get_info() -> MethodInfo:
        return MethodInfo(parallel=True, iterations=None)

    @staticmethod
    @abstractmethod
    def get_hyperopt_space() -> Any:
        ...

    @staticmethod
    @abstractmethod
    def get_pipeline() -> Any:
        ...


class IndexAccess(ABC):

    @abstractmethod
    def __getitem__(self, key):
        ...

    @abstractmethod
    def __setitem__(self, key, value):
        ...


CVSampling = Iterable[Tuple[List[int], List[int]]]


class NestedTrainSet(TypedDict):
    cv: Iterable[Tuple[List[int], List[int]]]
    all: Iterable[int]


NestedCV = List[Tuple[NestedTrainSet, List[int]]]


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
    payload: Dict
