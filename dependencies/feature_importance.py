from typing import List, Callable, Any

from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from pandas import DataFrame, Series
from toolz.curried import map

from dependencies.evaluation_functions import join_repeats_and_folds_cv_results
from dependencies.custom_types import ModelCVResult
from dependencies.functional import pipe


def plot_feature_importance_formatted(
    results: List[ModelCVResult],
    format_feature: Callable[[str], str],
    n_features: int = None,
    axis: Axis = None,
    bar_color_callback: Callable[[str], str] = None,
):
    _feature_importance = join_repeats_and_folds_cv_results(results)['feature_importance']

    _feature_importance_formatted = DataFrame(
        {
            'importance': _feature_importance['mean'],
            'name': list(
                map(
                    # ' ' hack: so we can display the same feature name for both donor and recipient
                    lambda feature: (' ' if 'don' in feature else '') + format_feature(feature),
                    _feature_importance.index
                )
            ),
        },
    )

    if n_features:
        _feature_importance_formatted = _feature_importance_formatted.iloc[:n_features]
        print(len(_feature_importance_formatted))

    plot_feature_importance(
        _feature_importance_formatted, axis=axis, bar_color_callback=bar_color_callback
    )


def plot_feature_importance(
    feature_importance_data: DataFrame,
    axis: Axis = None,
    bar_color_callback: Callable[[str, Series], str] = None,
) -> None:

    target = (axis if axis else plt)

    feature_importance_data = feature_importance_data.iloc[
        feature_importance_data['importance'].abs().argsort()]

    target.margins(y=0.01, x=0.01)

    max_feature_importance = max(series.iloc[0] for _, series in feature_importance_data.iterrows())

    for identifier, row in feature_importance_data.iterrows():

        color = bar_color_callback(identifier, row) if bar_color_callback else '#377eb8'

        bar = target.barh(
            row['name'],
            row['importance'],
            color=color,
            zorder=2,
        )

        for rect in bar:
            target.text(
                (max_feature_importance * 1.055),
                rect.get_y() + 0.2,
                str('{:<05}'.format(round(row['importance'], 3))),
                ha='left',
                fontsize=13,
            )

    target.set_yticklabels(feature_importance_data['name'], fontsize=12)


def label_bars(rects, feature_importance_data, axis: Any) -> None:

    max_width = pipe(
        rects,
        map(lambda rect: rect.get_width()),
        max,
    )

    for index, rect in enumerate(rects):
        try:
            order_std = " (pos. ± " + str(
                round(feature_importance_data.iloc[index]['order std'], 2)
            ) + ")"
        except KeyError:
            order_std = ""

        axis.text(
            max_width + 0.01,
            rect.get_y() + 0.2,
            str(round(feature_importance_data.iloc[index]['importance'], 3)) + order_std,
            ha='left',
            fontsize=13,
        )


def savefig(name):
    plt.savefig(f'./data/plots/{name}')
    plt.savefig(f'/home/sitnarf/vbshared/{name}')
