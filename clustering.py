import logging
import math
import multiprocessing
from functools import partial
from itertools import combinations, repeat
from logging import warning
from random import randint
# noinspection Mypy
from typing import List, Dict, TypedDict, Tuple, Optional, Any

import numpy as np
from numpy import std, ndarray
from pandas import DataFrame, Series
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial.distance import hamming
from scipy.stats import chi2_contingency
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, MeanShift, OPTICS, Birch, \
    AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.weightstats import ztest
from toolz.curried import map, sorted, get

from custom_types import Estimator, ClusteringEstimator
from formatting import format_p_value, format_count_and_percentage
from functional import decorate_unpack, try_except, statements, map_tuples, find, pipe
from methods.methods_utils import get_categorical_features
from statistics_functions import round_digits
from utils import RandomSeed, from_items


class ClusteringInternalMetrics(TypedDict):
    si: float
    dbi: float


class ClusteringExternalMetrics(TypedDict):
    # purity: float
    # average_purity: float
    average_gini_impurity: float


class ClusteringMetrics(ClusteringInternalMetrics, ClusteringExternalMetrics):
    pass


class ClusteringProtocol:
    identifier: Optional[str]
    title: Optional[str]
    distance_metric: str
    parameters: Dict

    def __init__(
        self,
        identifier: str = None,
        title: str = None,
        distance_metric: str = 'sqeuclidean',
        parameters=None
    ):
        self.distance_metric = distance_metric
        self.title = title
        self.identifier = identifier
        self.parameters = parameters if parameters else {}

    def get_si_score(self, X: DataFrame, y_pred: Series) -> float:
        return silhouette_score(X, y_pred, metric=self.distance_metric)

    @staticmethod
    def get_calinski_harabasz(X: DataFrame, y_pred: Series) -> float:
        return calinski_harabasz_score(X, y_pred)

    @staticmethod
    def get_db_index(X: DataFrame, y_pred: Series) -> float:
        return davies_bouldin_score(X, y_pred)

    def measure_internal_metrics(self, X: DataFrame, y_pred: Series) -> ClusteringInternalMetrics:
        try:
            si = self.get_si_score(X, y_pred)
        except ValueError:
            si = math.nan

        try:
            dbi = self.get_db_index(X, y_pred)
        except ValueError:
            dbi = math.nan

        return ClusteringInternalMetrics(
            si=si,
            dbi=dbi,
        )

    @staticmethod
    def measure_external_metrics(y_pred: List, y_true: List[int]) -> ClusteringExternalMetrics:
        return ClusteringExternalMetrics(
            # purity=purity_score(y_true, y_pred),
            # average_purity=average_purity(y_pred, y_true),
            average_gini_impurity=average_gini_impurity(y_pred, y_true),
        )

    def measure_metrics(
        self, X: DataFrame, y_pred: List[int], y_true: List[int]
    ) -> ClusteringMetrics:
        return ClusteringMetrics(
            **self.measure_internal_metrics(X, y_pred),
            **self.measure_external_metrics(y_pred, y_true)
        )

    def algorithm(self, X: DataFrame, n_cluster: int) -> List[int]:
        raise NotImplemented()


def get_cluster_identifiers(y_pred: Series) -> List[int]:
    return list(np.unique(y_pred.to_list()))


def get_cluster_count(y_pred: List):
    return len(get_cluster_identifiers(y_pred))


def get_instances_per_cluster(X: DataFrame, y_pred: Series) -> List[DataFrame]:
    labels = get_cluster_identifiers(y_pred)
    return [X[y_pred == label] for label in labels]


def count_values_and_align(series1: Series, series2: Series) -> Tuple[Series, Series]:
    values1 = series1.copy().value_counts().sort_index()
    values2 = series2.copy().value_counts().sort_index()
    indexes = pipe(
        set(values1.index).union(set(values2.index)),
        list,
        sorted,
    )
    for values in (values1, values2):
        for index in indexes:
            try:
                values.loc[index]
            except KeyError:
                values.loc[index] = 0

    return values1.sort_index(), values2.sort_index()


def measure_cluster_features_statistics(X: DataFrame, y_pred: Series):
    X = X.copy()

    # try:
    #     X['SMK'] = X['SMK'] >= 1
    #     X['SMK'] = X['SMK'].apply(lambda value: 1 if value else 0)
    # except KeyError:
    #     pass
    log_transformed = ('LPRA', 'LINS', 'LLEPT', 'LFERR', 'LALDO', 'LCRTSL')

    for feature in log_transformed:
        if feature not in X:
            logging.warning(f'Feature {feature} not present')
            continue

        X[feature] = 10**X[feature]

    non_normal_features = (
        'LFERR', 'LGGT', 'SS', 'LFERR', 'LPRA', 'LINS', 'LLEPT', 'LALDO', 'LCRTSL', 'SA_V3',
        'RA1_AVL'
    )

    try:
        del X['DBIRTH']
    except KeyError:
        pass

    try:
        del X['DATT']
    except KeyError:
        pass

    try:
        del X['SFILE']
    except KeyError:
        pass

    labels = get_cluster_identifiers(y_pred)
    X_clustered = [X[y_pred == label] for label in labels]
    categorical_features = get_categorical_features(X)

    continuous_features = [column for column in X.columns if column not in categorical_features]
    data_frame = DataFrame(index=X.columns)
    rename_features = {}

    for cluster_index, cluster in enumerate(X_clustered):
        cluster_feature_statistics = {}

        for feature in categorical_features:
            if feature == 'SOCK':
                total = len(cluster[feature].dropna())
                classes = sorted(cluster[feature].unique())
                value = '/'.join(
                    [
                        f'{round((len(cluster[cluster[feature] == class_value]) / total) * 100, 1)}'
                        for class_value in classes
                    ]
                )
                rename_features[feature] = feature
            else:
                value_count = cluster[feature].value_counts()
                if len(value_count) < 2:
                    warning(f'Skipped feature {feature}')
                    continue
                value = format_count_and_percentage(value_count, decimals=1)

            cluster_feature_statistics[feature] = value

        for column in continuous_features:
            mean_value = float(cluster[column].mean())

            if column in non_normal_features:
                spread_statistic = f' ({round(cluster[column].quantile(0.1), 2)}' \
                                   f'-{round(cluster[column].quantile(0.9), 2)})'
            else:
                spread_statistic = f' ± {round(std(cluster[column], ddof=1), 3)}'

            cluster_feature_statistics[column] = str(round_digits(mean_value, 3)) + spread_statistic

        cluster_column_key = f'cluster {cluster_index + 1} (n={len(cluster)})'
        data_frame[cluster_column_key] = Series(cluster_feature_statistics)

    for cluster1, cluster2 in combinations(range(len(X_clustered)), 2):
        # noinspection PyUnresolvedReferences
        continuous_statistics = {
            column: format_p_value(
                ztest(
                    X_clustered[cluster1][column].dropna(),
                    X_clustered[cluster2][column].dropna(),
                )[1]
            )
            for column in continuous_features
        }
        # sklearn.feature_selection.chi2(DataFrame, y)
        categorical_statistics = {
            column: format_p_value(
                chi2_contingency(
                    count_values_and_align(
                        X_clustered[cluster1][column], X_clustered[cluster2][column]
                    ),
                    correction=False,
                )[1]
            )
            for column in categorical_features
        }

        data_frame[f'p value {cluster1} ⇄ {cluster2} (95 %)'] = Series(
            {
                **categorical_statistics,
                **continuous_statistics
            }
        )
    missing_values = Series(X.isnull().sum(), index=data_frame.index)
    data_frame = data_frame.assign(missing=missing_values)
    data_frame.rename(index=rename_features, inplace=True)

    return data_frame


def get_counts_per_cluster(y_pred: Series, y_true: Series) -> Dict[int, Series]:
    cluster_labels = get_cluster_identifiers(y_pred)
    true_labels = get_cluster_identifiers(y_true)
    counts = {}
    for cluster_label in cluster_labels:
        index_cluster = y_pred == cluster_label
        value_counts = y_true[index_cluster].value_counts()
        for true_label in true_labels:
            if true_label not in value_counts:
                value_counts[true_label] = 0
        counts[cluster_label] = value_counts.sort_index()
    return counts


def purity_score(y_true, y_pred):
    contingency_matrix_result = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result)


def gini_impurity(points: Dict[int, int]) -> float:
    gi = 1.0
    total = sum(points)
    for label_count in points:
        gi -= (label_count / total)**2
    return gi


def average_gini_impurity(y_pred: List[int], y_true: List[int]):
    clusters = get_counts_per_cluster(y_pred, y_true)
    rates = []
    total_count = len(y_pred)
    for cluster_index, cluster_counts in clusters.items():
        total_cluster_count = cluster_counts.sum()
        gi = gini_impurity(cluster_counts.to_dict().values())
        gi *= total_cluster_count / total_count
        rates.append(gi)
    return np.sum(rates)


def average_purity(y_pred: List[int], y_true: List[int]) -> float:
    clusters = get_counts_per_cluster(y_pred, y_true)
    rates = []
    for cluster_index, cluster_counts in clusters.items():
        majority_class = cluster_counts.idxmax()
        majority_count = cluster_counts[majority_class]
        total_count = cluster_counts.sum()
        rate = majority_count / total_count
        rates.append(rate)
    return np.mean(rates)


def optimal_k(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference benchmark to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)), ))
    resultsdf = DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (
        gaps.argmax() + 1, resultsdf
    )  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


class KMeansProtocol(ClusteringProtocol):

    @staticmethod
    def get_pipeline(n_clusters: int) -> Pipeline:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('clustering', KMeans(
                    init='random',
                    n_jobs=-1,
                    n_clusters=n_clusters,
                )),
            ]
        )

    def algorithm(self, X, n_clusters) -> List[int]:
        return self \
            .get_pipeline(n_clusters).set_params(**self.parameters) \
            .fit_predict(X)


# class WKKMeansProtocol(ClusteringProtocol):
#
#     @staticmethod
#     def get_pipeline(k: int) -> Pipeline:
#         return Pipeline([
#             ('scaler', StandardScaler()),
#             ('clustering', WKMeans(k, beta=1)),
#         ])
#
#     def algorithm(self, X, n_clusters) -> List[int]:
#         return self \
#             .get_pipeline(n_clusters).set_params(**self.parameters) \
#             .fit_predict(X)


class AgglomerativeProtocol(ClusteringProtocol):

    @staticmethod
    def get_pipeline(n_clusters) -> Estimator:
        return Pipeline(
            [
                ('scaler', StandardScaler()),
                ('clustering', AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward',
                )),
            ]
        )

    def algorithm(self, X, n_clusters) -> List[int]:
        return self \
            .get_pipeline(n_clusters).set_params(**self.parameters) \
            .fit_predict(X)


class KMeansDRProtocol(ClusteringProtocol):

    def __init__(self, *args, dim_reduction_method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_reduction_method = dim_reduction_method or PCA(n_components=3)

    def algorithm(self, X, n_clusters) -> List[int]:
        clusterer = make_pipeline(
            # StandardScaler(),
            self.dim_reduction_method,
            KMeans(
                **{
                    **dict(
                        n_clusters=n_clusters,
                        init='random',
                        n_jobs=-1,
                    ),
                    **self.parameters,
                }
            ),
        )
        y_pred = clusterer.fit_predict(X)
        return y_pred


class KMeansTSNEProtocol(ClusteringProtocol):

    def __init__(self, *args, dim_reduction_method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_reduction_method = dim_reduction_method or PCA(n_components=3)

    def algorithm(self, X, n_clusters) -> List[int]:
        X = TSNE().fit_transform(X)
        clusterer = make_pipeline(
            # StandardScaler(),
            KMeans(
                **{
                    **dict(
                        n_clusters=n_clusters,
                        init='random',
                        n_jobs=-1,
                    ),
                    **self.parameters,
                }
            ),
        )
        y_pred = clusterer.fit_predict(X)
        return y_pred


class KMedoidsProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        from sklearn_extra.cluster import KMedoids
        clusterer = make_pipeline(
            StandardScaler(),
            KMedoids(
                **{
                    **dict(
                        n_clusters=n_clusters,
                        metric=self.distance_metric,
                        max_iter=1000,
                    ),
                    **self.parameters,
                }
            )
        )
        y_pred = clusterer.fit_predict(X)
        return y_pred


class GaussianMixtureProtocol(ClusteringProtocol):

    @staticmethod
    def get_pipeline(k: int) -> Pipeline:
        # noinspection PyArgumentEqualDefault
        return Pipeline(
            [
                (
                    'clustering',
                    GaussianMixture(
                        n_components=k,
                        covariance_type='full',
                        n_init=50,
                    )
                ),
            ]
        )

    def get_pipeline_with_params(self, n_clusters):
        return self.get_pipeline(k=n_clusters).set_params(**self.parameters)

    def get_bic(self, X: DataFrame, n_clusters: int) -> float:
        pipeline = self.get_pipeline_with_params(n_clusters)
        pipeline.fit(X)
        return pipeline[-1].bic(X)

    def algorithm(self, X: DataFrame, n_clusters: int) -> List[int]:
        pipeline = self.get_pipeline_with_params(n_clusters)
        y_pred = pipeline.fit_predict(X)
        return y_pred


class BayesianGaussianMixtureProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        clusterer = make_pipeline(
            BayesianGaussianMixture(**{
                **dict(n_components=n_clusters, ),
                **self.parameters,
            })
        )
        y_pred = clusterer.fit_predict(X)
        return y_pred


class SpectralProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        clusterer = make_pipeline(
            StandardScaler(),
            SpectralClustering(
                **{
                    **dict(
                        n_clusters=n_clusters,
                        # eigen_solver='arpack',
                        affinity="nearest_neighbors",
                        n_jobs=-1,
                    ),
                    **self.parameters,
                }
            )
        )
        y_pred = clusterer.fit_predict(X)
        return y_pred


class AffinityPropagationProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        method = make_pipeline(
            StandardScaler(),
            AffinityPropagation(**{
                **dict(),
                **self.parameters,
            }),
        )
        y_pred = method.fit_predict(X)
        return y_pred


class MeanShiftProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        method = make_pipeline(
            StandardScaler(),
            MeanShift(**{
                **dict(),
                **self.parameters,
            }),
        )
        y_pred = method.fit_predict(X)
        return y_pred


class OpticsProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        method = make_pipeline(
            StandardScaler(),
            OPTICS(**{
                **dict(),
                **self.parameters,
            }),
        )
        y_pred = method.fit_predict(X)
        return y_pred


class BirchProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        method = make_pipeline(
            StandardScaler(),
            Birch(**{
                **dict(n_clusters=n_clusters),
                **self.parameters,
            }),
        )
        y_pred = method.fit_predict(X)
        return y_pred


class HDBSCANProtocol(ClusteringProtocol):

    def algorithm(self, X, n_clusters) -> List[int]:
        from hdbscan import HDBSCAN
        clusterer = make_pipeline(StandardScaler(), HDBSCAN())
        y_pred = clusterer.fit_predict(X)
        return y_pred


def get_cluster_mapping_by_prevalence(
    y_pred: Series,
    y_true: Series,
) -> Dict:

    def get_1_prevalence(distribution: Series) -> int:
        try:
            prevalence_1 = (distribution[1] / distribution.sum())
        except KeyError:
            prevalence_1 = 0
        return prevalence_1

    return pipe(
        get_counts_per_cluster(y_pred, y_true),
        dict.items,
        partial(map_tuples, lambda index, distribution: (index, get_1_prevalence(distribution))),
        sorted(key=get(1)),
        enumerate,
        partial(map_tuples, lambda index, item: (item[0], index)),
        list,
        from_items,
    )


def sort_y_proba_by_prevalence(y_proba: DataFrame, y_true: Series) -> DataFrame:
    y_proba_new = y_proba.copy()

    y_pred: Series = get_y_pred_from_y_proba(y_proba)

    class_mapping = get_cluster_mapping_by_prevalence(y_pred, y_true)

    for from_class, to_class in class_mapping.items():
        y_proba_new[to_class] = y_proba[from_class]

    y_proba_new_reindexed = y_proba_new.reindex(sorted(y_proba_new.columns), axis=1)
    return y_proba_new_reindexed


def get_y_pred_from_y_proba(y_proba: DataFrame) -> Series:
    return pipe(
        y_proba.iterrows(),
        partial(
            map,
            decorate_unpack(
                lambda _, row: statements(
                    max_value := max(list(row.values)),
                    find(lambda value: value[1] == max_value, row.items())[0],
                )
            )
        ),
        partial(Series, index=y_proba.index),
    )


def map_y_pred_by_prevalence(
    y_pred: Series,
    y_true: Series,
) -> Series:
    y_pred_new = y_pred.copy()
    class_mapping = get_cluster_mapping_by_prevalence(y_pred, y_true)
    for from_class, to_class in class_mapping.items():
        y_pred_new[y_pred == from_class] = to_class
    return y_pred_new


def stability_index(
    k: int,
    X: DataFrame,
    clustering_estimator: Any,
    classifier_estimator: Any,
    repeats: int = 10,
    n_jobs: int = 10,
) -> Tuple[float, float]:
    with multiprocessing.Pool(n_jobs) as pool:
        values = pool.starmap(
            stability_index_once,
            zip(
                repeat(k),
                repeat(X),
                repeat(clustering_estimator),
                repeat(classifier_estimator),
                range(repeats),
            )
        )

    return float(np.mean(values)), float(np.std(values))


def stability_index_once(
    k: int,
    X: DataFrame,
    clustering_estimator: ClusteringEstimator,
    classifier_estimator: Estimator,
    random_state: int = None,
) -> float:
    with RandomSeed(random_state):
        X_train, X_test = train_test_split(X, test_size=0.5, random_state=random_state)
        y_train = clustering_estimator.fit_predict(X_train)
        y_test = clustering_estimator.fit_predict(X_test)

        classifier_estimator.fit(X_train, y_train)
        y_pred = classifier_estimator.predict(X_test)

        aligned_labels, aligned_pred_labels = align_clusters(y_test, y_pred)

        predicted_accuracy = 1 - hamming(aligned_labels, aligned_pred_labels)

        y_pred_rand_1 = generate_random_clusters(k, size=len(y_train))
        y_pred_rand_2 = generate_random_clusters(k, size=len(y_train))

        rand_accuracy = 1 - hamming(y_pred_rand_1, y_pred_rand_2)

        return predicted_accuracy / rand_accuracy


class ClusterClassMismatch(Exception):
    ...


def supply_missing_labels(contingency: ndarray) -> ndarray:
    max_dimension = max(contingency.shape)
    return np.array(
        [
            np.array(
                [
                    try_except(lambda: contingency[x, y], {Exception: lambda: 0})
                    for y in range(max_dimension)
                ]
            )
            for x in range(max_dimension)
        ]
    )


def align_clusters(y_pred_1: List[int], y_pred_2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Use the Hungarian (Kuhn–Munkres) algorithm to align two sets of cluster
    labels for the same dataset as phrased as a combinational optimziation
    algorithm that runs in O(n^3) time. Recreates the cluster labels for
    both versions of the cluster labels and returns them.
    """

    contingency, labels_v1, labels_v2 = contingency_matrix(y_pred_1, y_pred_2)
    contingency = supply_missing_labels(contingency)
    contingency = contingency * -1
    assignments = linear_sum_assignment(contingency)

    new_labels_v1 = [assignments[0][label] for label in y_pred_1]
    new_labels_v2 = [assignments[1][label] for label in y_pred_2]

    return new_labels_v1, new_labels_v2


def contingency_matrix(
    y_pred_1: List[int],
    y_pred_2: List[int],
    eps: Optional[float] = None,
) -> Tuple[ndarray, List[int], List[int]]:
    classes, class_idx = np.unique(y_pred_1, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred_2, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=np.int
    ).toarray()
    if eps is not None:
        contingency = contingency + eps
    return contingency, class_idx, cluster_idx


def generate_random_clusters(n_clusters: int, size: int) -> List[int]:
    """
    Generate two random sets of cluster labels with n_clusters
    with an equal 1/n_clusters chance for each point to belong
    to a given cluster.
    """
    rand_labels = [randint(1, n_clusters) for _ in range(size)]
    return rand_labels
