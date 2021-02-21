from functools import partial
from sys import argv

import pandas
from functional_pipeline import pipeline, flatten
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from tabulate import tabulate
from toolz.curried import map
from typing import List, Callable, Tuple

from evaluation_functions import ModelCVResult, join_repeats_and_folds_cv_results
from functional import pipe
from utils import map_index


def label_bars(rects, feature_importance):
    max_width = pipe(
        rects,
        map(lambda rect: rect.get_width()),
        max,
    )

    for index, rect in enumerate(rects):
        try:
            order_std = " (pos. ± " + str(
                round(feature_importance.iloc[index]['order std'], 2)
            ) + ")"
        except KeyError:
            order_std = ""

        plt.text(
            max_width + 0.01,
            rect.get_y() + 0.2,
            str(round(feature_importance.iloc[index]['mean'], 3)) + order_std,
            ha='left',
        )


def plot_feature_importance(feature_importance_mean_df):
    feature_importance_mean_df = feature_importance_mean_df.iloc[feature_importance_mean_df['mean'].abs().argsort()]
    plt.margins(y=0.01)
    bar = plt.barh(
        feature_importance_mean_df.index,
        feature_importance_mean_df['mean'],
        color="r",
    )

    label_bars(bar, feature_importance_mean_df)


def plot_feature_importance_formatted(
        results: List[ModelCVResult],
        format_feature: Callable[[str], str],
        n_features: int = None):
    _feature_importance = join_repeats_and_folds_cv_results(results)[
        'feature_importance']
    _feature_importance = map_index(format_feature, _feature_importance)
    if n_features:
        _feature_importance = _feature_importance.iloc[:n_features]
    plot_feature_importance(_feature_importance)


def savefig(name):
    plt.savefig(f'./data/plots/{name}')
    plt.savefig(f'/home/sitnarf/vbshared/{name}')
