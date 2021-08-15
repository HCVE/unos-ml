import re
from functools import partial
from numbers import Real
from typing import Any, Dict, Tuple, Callable, Iterable, Mapping, TypeVar, List, Set, Union

from matplotlib import pyplot
from toolz import identity

from custom_types import ClassificationMetrics, ClassificationMetricsWithStatistics
from include.functional import pipe, flatten, statements, compact
from include.utils import get_class_attributes, capitalize_first, decompose_one_hot_feature_name

ALL_METRICS = get_class_attributes(ClassificationMetrics)
T = TypeVar('T')


def p(text: str) -> None:
    from IPython.core.display import display, HTML
    display(HTML(f'<p>{text}</b>'))


def format_ci(ci: Tuple) -> str:
    try:
        return f'{format_decimal(ci[0])}–{format_decimal(ci[1])}'
    except (ValueError, TypeError):
        return ''


def format_decimal(number: float) -> str:
    return str(round(number * 1000) / 1000)


def format_structure(formatter: Callable, structure: Any) -> Any:
    if isinstance(structure, Iterable) and (not isinstance(structure, str)):
        return [format_structure(formatter, item) for item in structure]
    elif isinstance(structure, Mapping):
        return {key: format_structure(formatter, value) for (key, value) in structure.items()}
    else:
        try:
            return formatter(structure)
        except Exception:
            return structure


def compare_metrics_in_table(
    metrics_for_methods: Dict[str, ClassificationMetricsWithStatistics],
    methods_order: List[str] = None,
    include: Tuple[str, ...] = ('balanced_accuracy', 'roc_auc', 'recall', 'fpr'),
    format_method_name: Callable[[str], str] = identity,
    include_ci_for: Set[str] = None,
    include_delta: bool = False,
    ci_in_separate_cell: bool = True,
) -> List[List]:
    if include_ci_for is None:
        include_ci_for = set(include)

    def get_line(
        method: str, metrics: Union[ClassificationMetrics, ClassificationMetricsWithStatistics]
    ):
        return [
            format_method_name(method),
            *pipe(
                include,
                partial(
                    map,
                    lambda metric: statements(
                        ci := format_ci(metrics[metric].ci),
                        [
                            format_decimal(metrics[metric].mean) + (
                                f' ({ci})'
                                if metric in include_ci_for and not ci_in_separate_cell else ''
                            ),
                            (
                                metrics[metric].mean -
                                get_max_metric_value(metric, metrics_for_methods.values())
                            ) if include_delta else None,
                        ] + ([ci] if metric in include_ci_for and ci_in_separate_cell else []),
                    ),
                ),
                flatten,
                compact,
            ),
        ]

    def get_max_metric_value(
        metric: str, results: Iterable[ClassificationMetricsWithStatistics]
    ) -> Real:
        return max(metrics[metric].mean for metrics in results)

    if not methods_order:
        methods_order = pipe(
            metrics_for_methods,
            partial(sorted, key=lambda i: i[1]['roc_acu'].mean),
            partial(map, lambda i: i[0]),
        )

    lines = [get_line(method, metrics_for_methods[method]) for method in methods_order]

    return format_structure(
        format_decimal,
        [
            [
                '', *flatten(
                    map(
                        lambda metric:
                        [format_metric_short(metric), *(['Δ'] if include_delta else [])] +
                        (['95% CI']
                         if metric in include_ci_for and ci_in_separate_cell else []), include
                    )
                )
            ],
            *lines,
        ],
    )


def format_metric_short(metric: str) -> str:
    try:
        return {
            'balanced_accuracy': 'BACC',
            'roc_auc': 'ROC AUC',
            'recall': 'TPR',
            'precision': 'PREC',
            'fpr': 'FPR',
            'tnr': 'TNR',
            'average_precision': 'AP',
            'brier_score': 'Brier',
        }[metric]
    except KeyError:
        return metric


def render_struct_table(table: List[List]) -> str:
    output = "<table>\n"

    for line in table:
        output += "<tr>\n"
        for cell in line:
            output += f"<td>{cell}</td>\n"
        output += "</tr>\n"

    return output + "</table>"


def set_style(ax=None) -> None:
    if ax is None:
        ax = pyplot.gca()
    pyplot.rcParams['axes.titlepad'] = 10
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_fontsize(13)
    ax.xaxis.labelpad = 5
    ax.yaxis.label.set_fontsize(13)
    ax.yaxis.labelpad = 7


def format_feature(
    metadata: Dict,
    feature_name: str,
    remove_donor_and_recipient=True,
    note_is_log: bool = True
) -> str:
    is_log = feature_name.startswith("log_")
    base_name = feature_name[len('log_'):] if is_log else feature_name
    decomposed_feature_name = decompose_one_hot_feature_name(base_name, metadata)

    try:
        metadata_item = metadata[decomposed_feature_name['name']]
        if remove_donor_and_recipient:
            final_base_name = pipe(
                metadata_item['name_long'],
                lambda s: re.sub(' ?recipient', '', s, flags=re.IGNORECASE),
                lambda s: re.sub(' ?donor', '', s, flags=re.IGNORECASE) if 'ECCT' not in s else s,
                str.strip,
                capitalize_first,
            )
        else:
            final_base_name = metadata_item['name_long']

        try:
            decomposed_value = metadata_item['value_map'][float(decomposed_feature_name["value"])]
        except (KeyError, TypeError, ValueError):
            decomposed_value = decomposed_feature_name["value"]

        return final_base_name + \
               (f' • {decomposed_value}' if decomposed_feature_name['value'] else '') + \
               ((' (log)' if is_log else '') if note_is_log else '')
    except KeyError:
        return feature_name


def format_heart_transplant_method_name(identifier: str) -> str:
    output = ""
    if 'survival_random_forest' in identifier:
        output = 'Random Survival Forest'
    elif 'random_forest' in identifier:
        output = 'Random Forest'
    elif 'survival_gradient_boosting' in identifier:
        output = 'Survival Gradient Boosting'
    elif 'l2_logistic_regression' in identifier:
        output = "L2 Logistic Regression"
    elif 'xgboost' in identifier:
        output = 'XGBoost'
    elif 'cox' in identifier:
        output = 'Cox regression'

    if 'no_imputation' in identifier:
        output += ' No Impute'

    return output
