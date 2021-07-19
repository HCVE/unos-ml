from numbers import Rational, Integral
from typing import Iterable, List, Dict, Any, Mapping, Union

import matplotlib.font_manager
import matplotlib.pyplot as plt
# import notify2
import numpy as np
import pandas as pd
from IPython.core.display import HTML
from IPython.display import display
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame, Series
from sklearn.metrics import roc_curve, auc
from toolz import merge
from toolz.curried import map

from custom_types import ModelCVResult, ModelResult
from evaluation_functions import get_result_vector_from_result, compute_threshold_averaged_roc
from formatting import format_item_label, format_style, format_item, Category, CategoryStyled, \
    Attribute, dict_to_table_vertical, dict_to_struct_table_horizontal, render_struct_table, \
    dict_to_struct_table_vertical
from functional import flatten, pipe
from utils import empty_dict


def display_number(i: Union[int, float]) -> None:
    display_html(format_number(i))


def format_number(i: Union[Integral, Rational, int]) -> str:
    return f'{i:,}'


def format_iterable(iterable: Iterable) -> str:
    return "\n".join(iterable)


def format_real_labels_distribution(distribution: Series) -> str:
    string = ', '.join((f'{label}: {value}' for label, value in distribution.items()))
    if len(distribution.keys()) == 2:
        string += f' ({(distribution[1] / distribution.sum()) * 100:.2f}%)'
    return string


def format_cluster_real_labels(statistic: Dict[int, Series]) -> str:
    return dict_to_table_vertical(
        {
            f'Cluster {cluster_index}': format_real_labels_distribution(distribution)
            for cluster_index, distribution in statistic.items()
        }
    )


def plot_roc_from_result_vector(
        y: Series,
        result: ModelResult,
        label: str = None,
        plot_kwargs: Mapping = None,
        display_random_curve: bool = True,
) -> None:
    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}

    fpr, tpr, _ = roc_curve(y.loc[result['y_test_score'].index], result['y_test_score'])
    auc_value = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        lw=1,
        label=f'{"ROC curve" if not label else label} (AUC=%0.3f)' % auc_value,
        **plot_kwargs
    )
    if display_random_curve:
        plt.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.75, linestyle='-')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_roc_from_result(
        y: Series,
        result: ModelCVResult,
        label: str = None,
        plot_kwargs: Mapping = None,
        display_random_curve: bool = True,
) -> None:
    plot_roc_from_result_vector(
        y,
        get_result_vector_from_result(result),
        label,
        plot_kwargs=plot_kwargs,
        display_random_curve=display_random_curve
    )


def plot_roc_from_results_averaged(
        y: Series,
        results: List[ModelCVResult],
        label: str = None,
        plot_kwargs: Mapping = empty_dict,
        display_random_curve: bool = True,
) -> None:
    normalized_fpr = np.linspace(0, 1, 99)

    def roc_curve_for_fold(y_score):
        fpr, tpr, thresholds = roc_curve(y.loc[y_score.index], y_score.iloc[:, 1])
        auc_value = auc(fpr, tpr)
        normalized_tpr = np.interp(normalized_fpr, fpr, tpr)
        return normalized_tpr, auc_value

    tprs: Any
    aucs: Any
    tprs, aucs = zip(
        *flatten(
            [[roc_curve_for_fold(y_score) for y_score in result['y_scores']] for result in results]
        )
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc: float = np.mean(aucs)
    std_auc: float = np.std(aucs, ddof=0)
    plt.plot(
        normalized_fpr, mean_tpr,
        **merge(
            dict(
                lw=1.5,
                label=f'{"ROC curve" if not label else label} (AUC=%0.3f)' % mean_auc,
            ),
            plot_kwargs,
        )
    )

    if display_random_curve:
        plt.plot([0, 1], [0, 1], color='#CCCCCC', lw=0.75, linestyle='-')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_roc_from_results_threshold_averaged(
        y_true: Series, y_scores: List[DataFrame], label: str = None
) -> None:
    lw = 2

    fpr, tpr, thresholds = compute_threshold_averaged_roc(y_true, y_scores)

    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, label=f'{"ROC curve" if not label else label}')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def print_fonts() -> None:
    def make_html(fontname):
        return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(
            font=fontname
        )

    code = "\n".join(
        [
            make_html(font)
            for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))
        ]
    )

    display(HTML("<div style='column-count: 2;'>{}</div>".format(code)))



def plot_feature_importance(coefficients: DataFrame, limit: int = None) -> None:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        if not limit:
            limit = len(coefficients)

        coefficients = coefficients.reindex(
            coefficients.abs().sort_values(ascending=True, by='mean').index
        )
        coefficients = coefficients[-limit:]

    plt.figure(figsize=(4, 7 * (limit / 25)))

    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
    )
    # plt.tick_params(axis='x', labelcolor='#414141', color='#b9b8b9')

    rects = plt.barh(
        coefficients.index,
        coefficients['mean'],
        color="#f89f76",
    )

    max_width = pipe(
        rects,
        map(lambda rect: rect.get_width()),
        max,
    )

    for index, rect in enumerate(rects):
        number = coefficients.iloc[index]['mean']
        plt.text(
            max_width * 1.1 + (-0.02 if number < 0 else 0),
            rect.get_y() + 0.2,
            f'{number:.3f}',
            # color='#060606',
            ha='left',
        )
    # plt.gcf().patch.set_facecolor('#fdeadd')
    plt.margins(y=0.01)
    # plt.gca().patch.set_facecolor('white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['right'].set_linewidth(1)
    plt.gca().spines['right'].set_color('#b9b8b9')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['left'].set_color('#b9b8b9')
    plt.gca().set_axisbelow(True)

    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 100

    plt.grid(axis='x')

    plt.gca().xaxis.grid(linestyle='--', which='major', linewidth=1)
    plt.gca().get_xgridlines()[1].set_linestyle('-')


def plot_style(grid_parameters: Dict = None, axis=None):
    rc('font', **{'family': 'Arial'})

    axis = axis or plt.gca()
    grid_parameters = grid_parameters or {}
    axis.grid(
        linestyle='--', which='major', color='#93939c', alpha=0.2, linewidth=1, **grid_parameters
    )
    axis.set_facecolor('white')

    for item in axis.spines.values():
        item.set_linewidth(1.4)
        item.set_edgecolor('gray')

    axis.tick_params(
        which='both',
        left=False,
        bottom=False,
        labelcolor='#314e5eff',
        labelsize=12,
    )

    axis.title.set_fontsize(15)
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')
    axis.xaxis.label.set_fontsize(14)
    axis.xaxis.labelpad = 5
    axis.yaxis.label.set_fontsize(14)
    axis.yaxis.labelpad = 7


def plot_line_chart(
        x,
        y,
        x_axis_label: str = None,
        y_axis_label: str = None,
        title: str = None,
        plot_parameters: Dict = None,
        grid_parameters: Dict = None,
        axis=None
):
    plot_parameters = plot_parameters or {}
    axis = axis or plt.gca()

    plot_style(axis=axis, grid_parameters=grid_parameters)

    plot = axis.plot(x, y, **plot_parameters)

    if x_axis_label:
        plt.xlabel(x_axis_label, labelpad=10)

    if y_axis_label:
        plt.ylabel(y_axis_label, labelpad=7)

    if title:
        plt.title(title)

    return plot


def display_print(content: Any) -> None:
    display(HTML(f'<pre>{content}</pre>'))


def savefig(*args, **kwargs) -> None:
    plt.savefig(
        *args,
        **merge(dict(bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white'), kwargs)
    )


italic = {'font-style': 'italic'}



@format_item.register(Category)
def __1(_item: Category, _phenogroups_comparison: DataFrame) -> str:
    return f'<td class="category" colspan="{len(_phenogroups_comparison.columns) + 1}">' + format_item_label(
        _item
    ) + '</td>'


@format_item.register(CategoryStyled)
def __2(_item: CategoryStyled, _phenogroups_comparison: DataFrame) -> str:
    return f'<td style="{format_style(_item.style)}" colspan="{len(_phenogroups_comparison.columns) + 1}">' + format_item_label(
        _item
    ) + '</td>'


@format_item.register(Attribute)
def __3(_item: Attribute, _phenogroups_comparison: DataFrame) -> str:
    _html = f'<td class="item">' + format_item_label(_item) + '</td>'
    phenogroups_comparison_row = _phenogroups_comparison.loc[_item.key]
    for value in phenogroups_comparison_row:
        _html += f'<td>{value}</td>'
    return _html



def list_of_lists_to_html_table(rows: List[List], style: str = None) -> str:
    html = '<table' + (f' style="{style}"' if style else '') + '>'
    for row in rows:
        html += '<tr>'
        for cell in row:
            html += f'<td>{cell}</td>'
        html += '</tr>'
    html += '</table>'
    return html


def format_cluster_features_statistics(statistics: DataFrame) -> DataFrame:
    new_statistics = statistics.copy()
    for column in statistics.columns:
        if column.lower().startswith('cluster'):
            new_statistics.rename(columns={column: column.capitalize()}, inplace=True)

        if column.lower().startswith('p value'):
            new_statistics.rename(
                columns={column: column.replace('p value', 'p-value')}, inplace=True
            )

        if column.lower().startswith('missing'):
            new_statistics.rename(
                columns={column: column.replace('missing', 'N missing values')}, inplace=True
            )

    return new_statistics


def set_integer_ticks(axis: Any = None) -> None:
    if not axis:
        axis = plt.gca().xaxis
    axis.set_major_locator(MaxNLocator(integer=True))


def fig_size(scale=1):
    size = plt.rcParams["figure.figsize"]
    size[0] = 30 * scale
    size[1] = 15 * scale
    plt.rcParams["figure.figsize"] = size


def display_html(html: str) -> None:
    # noinspection PyTypeChecker
    display(HTML(html))


def text_title(string: str) -> str:
    return string + '\n' + ('―' * len(string))


def text_main_title(string: str) -> str:
    return string + '\n' + '=' * len(string) + '\n'





def display_dict_as_table_horizontal(input_dict: Dict) -> None:
    pipe(
        input_dict,
        dict_to_struct_table_horizontal,
        render_struct_table,
        display_html,
    )


def display_dict_as_table_vertical(input_dict: Dict) -> None:
    pipe(
        input_dict,
        dict_to_struct_table_vertical,
        render_struct_table,
        display_html,
    )


def display_histogram(data_frame: DataFrame) -> None:
    data_frame.replace(np.nan, 'NAN').hist(grid=False)


def sort_legend(axis: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    axis.legend(handles, labels, loc='lower right')
