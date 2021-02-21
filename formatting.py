import textwrap
from dataclasses import dataclass
from functools import partial, singledispatch
from numbers import Real
from typing import Any, List, Dict, Tuple, Callable, Iterable, Mapping, Union, cast, TypeVar

from pandas import Series, DataFrame
from pyramda import keys
from tabulate import tabulate
from toolz.curried import get, pluck, map

from custom_types import ClassificationMetricsWithStatistics, ClassificationMetrics, GenericConfusionMatrix
from functional import flatten, pipe, find_index
from utils import get_class_attributes, get_tabulate_format, load_dictionary, get_feature_category

ALL_METRICS = get_class_attributes(ClassificationMetrics)
T = TypeVar('T')


def h1(text: str) -> None:
    from IPython.core.display import display, HTML
    # noinspection PyTypeChecker
    display((HTML(f'<h1>{text}</h1>')))


def h2(text: str) -> None:
    from IPython.core.display import display, HTML
    # noinspection PyTypeChecker
    display((HTML(f'<h2>{text}</h2>')))


def h3(text: str) -> None:
    from IPython.core.display import display, HTML
    # noinspection PyTypeChecker
    display((HTML(f'<h3>{text}</h3>')))


def h4(text: str) -> None:
    from IPython.core.display import display, HTML
    # noinspection PyTypeChecker
    display((HTML(f'<h4>{text}</h4>')))


def b(text: str) -> None:
    from IPython.core.display import display, HTML
    # noinspection PyTypeChecker
    display((HTML(f'<b>{text}</b>')))


def format_confidence_interval(m, interval):
    return str(round(m, 3)) + f' ({round(interval[0], 3)}—{round(interval[1], 3)})'


def format_percents(fraction: float, decimals: int = 1, sign: bool = True) -> str:
    return str(round(fraction * 100, decimals)) + ('%' if sign else '')


def format_p_value(fraction: float, max_digits: int = 4, min_digits=2) -> str:
    if fraction > 1 or fraction < 0:
        raise Exception('P-value can must be in range [0, 1]')

    if fraction == 1:
        return "1"

    rounded = round(fraction, max_digits)
    if rounded == 0:
        return f'<.{"0" * (max_digits - 1)}1'
    else:
        str_fraction = '%f' % fraction
        non_zero_index = find_index(
            cast(Callable[[T], bool], lambda value: str(value) != '.' and int(value) > 0),
            str_fraction,
        )
        output = str(fraction)
        order_correction = 10 ** (non_zero_index + (2 - min_digits))
        output = str(round(float(output) * order_correction) / order_correction)
        output = str(round(float(output), max_digits))
        return output + ("0" if len(output) == 3 else "")


def format_ci(ci: Tuple) -> str:
    return f'{format_decimal(ci[0])}—{format_decimal((ci[1]))}'


def format_count_and_percentage(distribution: Series, decimals: int = 0) -> str:
    distribution_sorted = distribution.copy().sort_index()
    total_number = distribution.sum()
    return " / ".join(
        [
            f'{subgroup_number} ({format_percents(subgroup_number / total_number, sign=False, decimals=decimals)})'
            for index, subgroup_number in enumerate(distribution_sorted)
            if index >= 1
        ]
    )


def format_decimal(number: float) -> str:
    return str(round(number * 1000) / 1000)


def format_structure(formatter: Callable, structure: Any) -> Any:
    if isinstance(structure, Iterable) and not isinstance(structure, str):
        return [format_structure(formatter, item) for item in structure]
    elif isinstance(structure, Mapping):
        return {key: format_structure(formatter, value) for key, value in structure.items()}
    else:
        # noinspection PyBroadException
        try:
            return formatter(structure)
        except Exception:
            return structure


def format_metrics_with_statistics(metrics: ClassificationMetricsWithStatistics) -> str:
    data = [
        [key, *value.format_to_list()] for key, value in
        (metrics.__dict__.items() if ('__dict__' in metrics) else metrics.items())
    ]
    return tabulate(data, tablefmt='fancy_grid')


def format_metric_short(metric: str) -> str:
    try:
        return {
            'balanced_accuracy': 'BACC',
            'roc_auc': 'ROC/AUC',
            'recall': 'TPR',
            'precision': 'PREC',
            'fpr': 'FPR',
            'tnr': 'TNR',
            'average_precision': 'AP',
        }[metric]
    except KeyError:
        return metric


def format_method(method: str) -> str:
    return method.split('.')[-1]


def format_method_nice(method: str) -> str:
    return method.split('.')[-1]


def compare_metrics_in_table(
        metrics_for_methods: Dict[str, ClassificationMetricsWithStatistics],
        include: Tuple[str, ...] = ('balanced_accuracy', 'roc_auc', 'recall', 'fpr'),
        include_ci: bool = False,
) -> List[List]:
    def get_line(method: str, metrics: Union[ClassificationMetrics, ClassificationMetricsWithStatistics]):
        return [
            method,
            *flatten([
                [
                    metrics[metric].mean,
                    metrics[metric].mean - get_max_metric_value(metric, metrics_for_methods.values()),
                ] + ([format_ci(metrics[metric].ci)] if include_ci else [])
                for metric in include
            ]
            )]

    lines = pipe(
        [get_line(method, metrics) for method, metrics in metrics_for_methods.items()],
        partial(sorted, key=get(1), reverse=True),
    )

    return format_structure(
        format_decimal,
        [
            ['', *flatten(
                map(lambda metric: [format_metric_short(metric), 'Δ'] + (['CI'] if include_ci else []), include))],
            *lines,
        ],
    )


def get_max_metric_value(metric: str, results: Iterable[ClassificationMetricsWithStatistics]) -> Real:
    return max(metrics[metric].mean for metrics in results)


def format_end_to_end_metrics_table(
        optimized_metrics_by_method: Dict[str, ClassificationMetricsWithStatistics],
        default_metrics_by_method: Dict[str, ClassificationMetricsWithStatistics],
        include_header: bool = True,
) -> List[List[str]]:
    methods = keys(optimized_metrics_by_method)

    def get_header() -> List[str]:
        return ['', 'Optimized ROC/AUC', 'Default ROC/AUC', 'Optimized PR/AUC', 'Default PR/AUC']

    def get_line(_method: str) -> List:
        _optimized_metrics = optimized_metrics_by_method[_method]
        _default_metrics = default_metrics_by_method[_method]
        return [
            format_method(_method),
            _optimized_metrics['roc_auc'],
            _default_metrics['roc_auc'],
            _optimized_metrics['average_precision'],
            _default_metrics['average_precision'],
        ]

    return [
        *([get_header()] if include_header else []),
        *pipe(
            [get_line(method) for method in methods],
            partial(sorted, key=lambda i: i[1].mean, reverse=True),
            map(lambda line: [
                line[0],
                *[cell.format_short() for cell in line[1:]],
            ]),
        ),
    ]


def format_single_threshold_table(
        metrics_by_method: Dict[str, ClassificationMetricsWithStatistics],
        include_header: bool = True,
) -> List[List[str]]:
    methods = keys(metrics_by_method)

    def get_line(_method: str) -> List:
        _metrics: ClassificationMetricsWithStatistics = metrics_by_method[_method]
        return [
            format_method(_method),
            _metrics['f1'].format_short(),
            _metrics['recall'].format_short(),
            _metrics['precision'].format_short(),
            _metrics['tnr'].format_short(),
            _metrics['npv'].format_short(),
        ]

    def get_header() -> List[str]:
        return [
            '', 'F1', 'Sensitivity (TPR)', 'Precision (PPV)', 'Specificity (TNR)',
            'Negative predictive value'
        ]

    return [
        *([get_header()] if include_header else []),
        *sorted((get_line(method) for method in methods), key=get(1), reverse=True),
    ]


def tabulate_formatted(*args, **kwargs) -> str:
    return tabulate(*args, **{**get_tabulate_format(), **kwargs})


def get_title(file):
    titles = {
        'adaboost': "Adaboost",
        'random_forest': "Random Forest",
        'xgboost': "XGBoost",
        'svm': "SVM",
        'logistic': "Logistic regression",
        'logitboost': "Logitboost",
    }
    title = ""
    for title_find, title_defined in titles.items():
        if title_find in file:
            title = title_defined

    if 'top' in file:
        title += ' top'

    if 'preselected' in file:
        title += ' preselected'

    if 'default' in file:
        title += ' (default)'

    return title


def dict_to_struct_table_horizontal(dictionary: Mapping) -> List[List]:
    items = list(dictionary.items())
    return [
        list(pluck(0, items)),
        list(pluck(1, items)),
    ]


def dict_to_struct_table_vertical(dictionary: Mapping) -> List[List]:
    return [list(item) for item in dictionary.items()]


def render_struct_table(table: List[List]) -> str:
    output = "<table>\n"

    for line in table:
        output += "<tr>\n"
        for cell in line:
            output += f"<td>{cell}</td>\n"
        output += "</tr>\n"

    return output + "</table>"


def dict_to_table_horizontal(dictionary: Mapping) -> str:
    return tabulate([[key, value] for key, value in dictionary.items()], **get_tabulate_format())


def dict_to_table_vertical(dictionary: Dict, digits: int = None) -> str:
    return tabulate(
        {
            key: [f'{value:.{digits}f}' if digits is not None else value]
            for key, value in dictionary.items()
        },
        headers='keys',
        **get_tabulate_format(),
    )


def format_list(iterable: Iterable) -> str:
    return ", ".join(iterable)


def format_iterable_vertical(iterable: Iterable) -> str:
    return "\n".join(iterable)


def confusion_matrix_to_table(matrix: GenericConfusionMatrix) -> List[List]:
    return [
        ['', 'Pred P', 'Pred N'],
        ['Real P', str(matrix.tp), str(matrix.fn)],
        ['Real N', str(matrix.fp), str(matrix.tn)],
    ]


def format_feature_short_readable(identifier: str) -> str:
    try:
        return {
            'EM': 'e\'',
            'AM': 'a\'',
            'SM': 's\'',
            'AGE': 'Age',
            'BMI': 'BMI',
            'SEX': 'Sex',
            'WHR': 'Waist/hip ratio',
            'RWT': 'LV RWT',
            'WAISTC': 'Waist circ.',
            'MCH': 'MCH',
            'HHT': "History of Htn",
            'TRT_HT': 'Anti-Htn treatment',
            'SOCK': 'Social class',
            'HCV2': 'CV disease',
            'HCHOL': 'HDL',
            'PP': 'Pulse pressure',
            'PR': 'Heart rate',
            'TRT_BB': 'β blockers',
            'TA_AVG': 'T wave ampl.',
            'RA1_V5': 'R wave ampl.',
            'SOKOLOW_LYON': 'Sokolow-Lyon',
            'LFERR': "Serum ferritin",
            'COF': 'Caffeine',
            'HTC': 'Haematocrit',
            'RA1_AVL': 'AVL R wave ampl.',
            'SBP': 'Systolic BP',
            'SKINF': 'Skinfold',
            'TRGL': 'Triglycerides',
            'LVMI': 'LV mass',
            'SV_MODI': 'LV stroke volume',
            'LA_EF_4CH': 'LA EF',
            'ESV_MODI': 'LV ESVi',
            'MV_DECT': 'MV dec. time',
            'EF_MOD': 'LV EF',
            'AO_DIAM': 'Aorta diameter',
            'MVA_VEL': 'A',
            'MVE_VEL': 'E',
            'LAESVI': 'LA ESVi',
            'REEM': 'E/e\'',
            'RMVEA': 'E/A',
            'REAM': 'e\'/a\'',
            'LA_GS': 'LA reservoir strain',
            'GS': 'LV global LS',
            'LVIDD': 'LV internal diameter',
            'LVPWD': 'LV wall thickness',
            'LA_ASI': 'LA Asi',
            'LAEDVI': 'LA EDVi',
            'LA_ADI': 'LA Adi',
        }[str(identifier).upper()]
    except KeyError:
        return identifier


def format_feature_medium_readable(identifier: str) -> str:
    try:
        return {
            'IVSD': 'IVS thickness',
            'LA_A_4CH': 'LA area change (4ch)',
        }[str(identifier).upper()]
    except KeyError:
        return format_feature_short_readable(identifier)


@dataclass
class Item:
    label: str
    indent: int


def format_item_label(_item: Item):
    return (_item.indent * 2 * "&nbsp;") + _item.label


def format_style(style: Dict) -> str:
    return pipe(
        (f'{key}: {value}' for key, value in style.items()),
        lambda _styles: ";".join(_styles),
    )


@singledispatch
def format_item(_1: Any, _2: Any) -> Any:
    raise NotImplementedError('Unsupported type')


def format_feature(feature_name):
    return format_feature_from_dictionary(feature_name, load_dictionary())


def format_feature_detailed(feature_name):
    try:
        return "[%s] %s" % (
            ",".join(get_feature_category(feature_name)), format_feature(feature_name)
        )
    except KeyError:
        return format_feature(feature_name)


def format_feature_from_dictionary(feature_name, dictionary):
    dictionary_capitalized_keys = {key.upper(): value for key, value in dictionary.items()}
    feature_name_capitalized = str(feature_name).upper()
    try:
        return "%s: %s" % (
            feature_name,
            textwrap.shorten(
                dictionary_capitalized_keys[feature_name_capitalized]["name"],
                width=50,
                placeholder="..."
            ),
        )
    except KeyError:
        return feature_name


def format_features_detailed(features):
    return "\n".join([format_feature(f) + (" [%s]" % get_feature_category(f)[0]) for f in features])


def compute_and_format_percents(amount: Union[float, int], amount_from: Union[float, int]) -> str:
    return f'{(amount / amount_from) * 100:.2f}%'


def format_columns_df(data_frame: DataFrame) -> DataFrame:
    data_frame_copy = data_frame.copy()
    data_frame_copy.columns = [format_feature_detailed(column) for column in data_frame_copy]
    return data_frame_copy


@dataclass
class Category(Item):
    ...


@dataclass
class CategoryStyled(Item):
    style: Dict


@dataclass
class Attribute(Item):
    key: str


def format_bytes(num: int) -> str:
    suffix = 'B'
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
