import os
import shelve
from functools import partial
from numbers import Rational, Integral
from os.path import basename
from typing import Union, Mapping, Callable, Dict, List, Set, Iterable, Any

import matplotlib.pyplot as plt
from IPython.core.display import display
from ipywidgets import HTML
from matplotlib import pyplot, pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pandas import Series, DataFrame
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from toolz import merge, identity, keyfilter
from toolz.curried import map

from include.custom_types import ModelCVResult, ModelResult, ValueWithStatistics, ClassificationMetricsWithStatistics
from include.evaluation_functions import compute_classification_metrics, join_repeats_and_folds_cv_results, \
    get_1_class_y_score, compute_auc_proc, join_folds_cv_result, get_cv_results_from_simple_cv_evaluation
from include.feature_importance import plot_feature_importance_formatted
from include.formatting import compare_metrics_in_table, render_struct_table, format_heart_transplant_method_name, p, \
    set_style, format_feature
from include.functional import pipe, or_fn, statements
from include.metadata import metadata
from include.utils import remove_suffix, decompose_one_hot_feature_name, get_feature_metadata


def format_number(i: Union[Integral, Rational, int]) -> str:
    return f'{i:,}'


def savefig(*args, **kwargs) -> None:
    plt.savefig(
        *args,
        **merge(dict(bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white'), kwargs)
    )


italic = {'font-style': 'italic'}


def display_html(html: str) -> None:
    # noinspection PyTypeChecker
    display(HTML(html))


def get_ht_metrics(
    results: Mapping[str, ModelCVResult],
    y: Series,
    filter_callback: Callable[[str], bool] = None,
    compute_metrics: Callable = None,
) -> Dict[str, Dict]:
    if compute_metrics is None:
        compute_metrics = compute_classification_metrics

    metrics: Dict = {}
    for name, item in results.items():
        if filter_callback is None or filter_callback(name):
            joined_result: ModelResult = join_repeats_and_folds_cv_results([item])
            metrics_item = compute_metrics(get_1_class_y_score(joined_result['y_test_score']), y)
            y_score = get_1_class_y_score(joined_result['y_test_score'])
            metrics[name] = {}
            metrics[name]['roc_auc'] = compute_auc_proc(y_score, y)
            metrics[name]['brier_score'] = ValueWithStatistics(
                metrics_item['brier_score'],
                std=None,
                ci=None,
            )
    return metrics


def get_ht_metrics_html_table(
    metrics: Dict[str, ClassificationMetricsWithStatistics],
    methods_order: List[str] = None,
    format_method_name: Callable[[str], str] = identity,
    include_ci_for: Set[str] = None,
    include_delta: bool = True,
) -> str:
    return pipe(
        compare_metrics_in_table(
            metrics,
            methods_order,
            include=('roc_auc', 'brier_score'),
            include_ci_for=include_ci_for,
            format_method_name=format_method_name,
            include_delta=include_delta,
            ci_in_separate_cell=False,
        ),
        render_struct_table,
    )


def display_and_save_tables(
    identifier_default: List[str], *args, identifier_tuned: List[str] = None, **kwargs
) -> None:
    if identifier_tuned is None:
        identifier_tuned = identifier_default

    html_table_params = dict(
        include_ci_for={'roc_auc'},
        include_delta=False,
        format_method_name=format_heart_transplant_method_name,
    )

    results_tuned = {}

    p('Optimized')
    for identifier in identifier_tuned:
        if not os.path.isfile(identifier):
            continue
        data = shelve.open(identifier, flag='r')
        results_tuned.update(get_results_from_data(data))
        data.close()

    tuned_table = get_ht_metrics(results_tuned, *args, filter_callback=tuned_filter, **kwargs)

    methods_order = pipe(
        tuned_table.items(),
        partial(sorted, key=lambda i: i[1]['roc_auc'].mean, reverse=True),
        partial(map, lambda i: i[0]),
        list,
    )
    tuned_html = get_ht_metrics_html_table(tuned_table, methods_order, **html_table_params)
    display_html(tuned_html)

    with open(f'{FIGURES_FOLDER}/tables/{basename(identifier_tuned[0])}_tuned.html', 'w') as f:
        f.write(tuned_html)

    try:
        p('Default')
        results_default = {}
        for identifier in identifier_default:
            if not os.path.isfile(identifier):
                continue
            data = shelve.open(identifier, flag='r')
            results_default.update(get_results_from_data(data))
            data.close()

        default_table = get_ht_metrics(
            results_default,
            *args,
            filter_callback=default_filter,
            **kwargs,
        )

        methods_order_default = pipe(
            methods_order,
            partial(map, lambda i: remove_suffix('_tuned', i)),
            list,
            lambda methods: methods + pipe(
                [
                    (method, metrics)
                    for method, metrics in default_table.items()
                    if remove_suffix('_default', method) not in methods
                ],
                partial(sorted, key=lambda i: i[1]['roc_auc'].mean, reverse=True),
                partial(map, lambda i: i[0]),
                list,
            ),
            partial(map, lambda i: i + '_default' if not i.endswith('_default') else i),
            partial(filter, lambda i: i in results_default.keys()),
            list,
        )

        default_html = get_ht_metrics_html_table(
            default_table, methods_order_default, **html_table_params
        )

        display_html(default_html)

        with open(
            f'{FIGURES_FOLDER}/tables/{basename(identifier_default[0])}_default.html', 'w'
        ) as f:
            f.write(default_html)

        data.close()
    except Exception:
        ...


def display_and_save_rocs(file_name: str, *args, title: str = None, **kwargs) -> None:
    figure = pyplot.figure(figsize=DEFAULT_FIGURE_SIZE)
    display_roc(file_name, figure, *args, **kwargs)
    if title:
        pyplot.title(title, pad=12)
        pyplot.gca().title.set_fontsize(15)
    save_roc(file_name)
    pyplot.show()


def display_roc(file_name, figure, *args, **kwargs):
    display_rocs_base(
        file_name,
        *args,
        filter_callback=or_fn(tuned_filter),
        style_by_callback=[color_methods_colorblind],
        format_method_name=format_heart_transplant_method_name,
        figure=figure,
        **kwargs,
    )


def display_rocs_base(
    file_name: str,
    y_true: Series,
    filter_callback: Callable[[str], bool],
    style_by_callback: Iterable[Callable[[str], Mapping]] = None,
    format_method_name: Callable[[str], str] = identity,
    figure: Figure = None,
    display_random_curve: bool = True,
) -> None:
    if not figure:
        pyplot.figure(figsize=DEFAULT_FIGURE_SIZE)

    style_by_callback = style_by_callback if style_by_callback is not None else {}

    results = shelve.open(file_name, flag='r')

    sorted_results = pipe(
        results, get_results_from_data, dict.items,
        partial(
            sorted,
            key=lambda i: compute_auc_proc(
                get_1_class_y_score(join_repeats_and_folds_cv_results([i[1]])['y_test_score']),
                y_true
            ).mean,
            reverse=True,
        )
    )

    for method_name, item in sorted_results:
        if filter_callback is None or filter_callback(method_name):
            style = merge(*[callback(method_name) for callback in style_by_callback], {})
            joined_result: ModelResult = join_repeats_and_folds_cv_results([item])
            y_score = get_1_class_y_score(joined_result['y_test_score'])
            auc = compute_auc_proc(y_score, y_true).mean
            fpr, tpr, _ = roc_curve(y_true.loc[y_score.index], y_score)
            pyplot.plot(
                fpr, tpr,
                **merge(
                    dict(
                        lw=1.5,
                        label=f'{format_method_name(method_name)} (AUC=%0.3f)' % auc,
                    ),
                    style,
                )
            )

    if display_random_curve:
        pyplot.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.75, linestyle='-')

    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend(loc="lower right")

    pyplot.gca().xaxis.label.set_fontsize(13)
    pyplot.gca().xaxis.labelpad = 5
    pyplot.gca().yaxis.label.set_fontsize(13)
    pyplot.gca().yaxis.labelpad = 7

    pyplot.grid(alpha=0.5, linestyle='--', linewidth=0.75)

    # try:
    #     sort_legend(pyplot.gca())
    # except ValueError:
    #     pass

    if not figure:
        pyplot.show()

    results.close()


def dashed_style(_: str) -> Mapping:
    return {'linestyle': '--'}


def display_calibration_plot(
    results: Mapping,
    y: Series,
    filter_callback: Callable[[str], bool] = None,
    style_by_callback: Iterable[Callable[[str], Mapping]] = None
) -> None:
    ax1 = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = pyplot.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    for method, item in sorted(results.items(), key=lambda i: i[0]):
        if filter_callback is None or filter_callback(method):
            style = merge(*[callback(method) for callback in style_by_callback], {})
            try:
                result = item['chosen']['result']
            except (KeyError, AttributeError):
                result = item

            result_joined: ModelResult = join_folds_cv_result(result)

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y.loc[result_joined['y_test_score'].index],
                get_1_class_y_score(result_joined['y_test_score']),
                n_bins=20
            )

            ax1.plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                label=('Default ' if method.endswith('_default') else '') +
                format_heart_transplant_method_name(method),
                **style,
            )

    set_style(ax1)
    ax1.legend(loc="lower right")
    ax1.set_ylabel("Fraction of positives")

    for method, item in results.items():
        if filter_callback is None or filter_callback(method):
            try:
                result = item['chosen']['result']
            except (KeyError, AttributeError):
                result = item
            result_joined: ModelResult = join_folds_cv_result(result)
            style = merge(*[callback(method) for callback in style_by_callback], {})
            ax2.hist(
                get_1_class_y_score(result_joined['y_test_score']),
                bins=20,
                label=method,
                histtype="step",
                lw=2,
                **style
            )
            set_style(ax2)
    ax1.grid(alpha=0.5, linestyle='--', linewidth=0.75)
    ax2.grid(alpha=0.5, linestyle='--', linewidth=0.75)
    ax2.set_xlabel("Predicted value")
    ax2.set_ylabel("Count")


def display_and_save_calibration_plot(
    file_names: List[str],
    *args,
    **kwargs,
) -> None:
    pyplot.figure(1, figsize=DEFAULT_FIGURE_SIZE)
    results = {}
    for file_name in file_names:
        with shelve.open(file_name) as data:
            for key, result in data.items():
                results[key] = result

    display_calibration_plot(results, *args, **kwargs)

    savefig(f'{FIGURES_FOLDER}/calibration/{basename(file_names[0])}.svg')
    pyplot.show()


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


def display_and_save_feature_importance(
    file_name: str,
    filter_callback: Callable[[str], bool] = None,
    n_features: int = 30,
    format_method_name: Callable[[str], str] = identity,
    method_order: List[str] = None,
) -> None:
    with shelve.open(file_name, flag='r') as results:
        results_filtered = keyfilter(
            lambda _name: filter_callback is None or filter_callback(_name), results
        )
        n_items = len(method_order)
        fig, ax = pyplot.subplots(1, n_items, figsize=(n_items * 6.3, 7))

        for i, name in enumerate(method_order):
            item = results[name]
            try:
                current_axis = ax[i]
            except TypeError:
                current_axis = ax

            current_axis.set_title(
                chr(65 + i) + '. ' + format_method_name(name),
                loc='center',
                pad=10,
                fontdict={'fontsize': 17}
            )

            if type(item) == dict and 'chosen' in item.keys():
                cv_results = get_cv_results_from_simple_cv_evaluation([item])
            else:
                cv_results = [item]

            colors_per_feature_type = {
                'recipient': '#377eb8',
                'donor': '#ff7f00',
                'matching': '#4daf4a',
            }
            plot_feature_importance_formatted(
                cv_results,
                format_feature=lambda feature_name: pipe(
                    feature_name,
                    partial(format_feature, metadata, note_is_log=False),
                ),
                n_features=n_features,
                axis=current_axis,
                bar_color_callback=lambda identifier, row: statements(
                    decomposed_feature_name := decompose_one_hot_feature_name(identifier, metadata),
                    colors_per_feature_type[
                        get_feature_metadata(decomposed_feature_name['name'], metadata)['type']]
                ),
            )

            current_axis.grid(
                linestyle='--', which='major', color='#93939c', alpha=0.6, linewidth=0.75, axis='y'
            )

            current_axis.legend(
                handles=[
                    Patch(color=color, label=feature_type.capitalize())
                    for feature_type, color in colors_per_feature_type.items()
                ],
                fontsize=13,
                loc='lower right',
            )
        pyplot.tight_layout()
        savefig(f'{FIGURES_FOLDER}/feature_importance/{basename(file_name)}.svg', dpi=300)


def save_roc(file_name):
    savefig(f'{FIGURES_FOLDER}/rocs/{basename(file_name)}.svg')


def color_methods(key: str) -> Mapping:
    colors = pyplot.get_cmap('Paired').colors
    if 'xgboost' in key:
        index = 1
    elif 'random_forest' in key:
        index = 3
    elif 'logistic' in key:
        index = 5
    else:
        index = 1

    if 'default' in key:
        index -= 1

    return {'color': colors[index]}


def color_methods_colorblind(key: str) -> Mapping:
    colors = [
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
        '#dede00'
    ]
    if 'xgboost' in key:
        index = 0
    elif 'survival_random_forest' in key:
        index = 3
    elif 'random_forest' in key:
        index = 1
    elif 'logistic' in key:
        index = 2
    elif 'survival_gradient' in key:
        index = 4
    else:
        index = 5

    return {'color': colors[index]}
