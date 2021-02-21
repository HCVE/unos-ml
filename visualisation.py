from typing import Iterable, List, Dict, Any, Mapping

import matplotlib.font_manager
import matplotlib.pyplot as plt
# import notify2
import numpy as np
import pandas as pd
from IPython.core.display import HTML
from IPython.display import display
from ipywidgets import widgets
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score, roc_curve, auc
from toolz.curried import map

from clustering import get_cluster_count, ClusteringProtocol, get_counts_per_cluster, \
    measure_cluster_features_statistics, map_y_pred_by_prevalence, get_cluster_identifiers
from evaluation_functions import ModelCVResult, ModelResult, \
    get_result_vector_from_result, compute_threshold_averaged_roc
from formatting import tabulate_formatted, format_item_label, format_style, format_item, Category, CategoryStyled, \
    Attribute, dict_to_table_vertical, dict_to_struct_table_horizontal, render_struct_table, \
    dict_to_struct_table_vertical
from functional import flatten, pipe


def init(percent: float = 60):
    notebook_width(percent)
    autostart()
    fix_scroll_bars()


def notebook_width(percent: float) -> None:
    display(
        HTML(
            data=f"""
                <style>
                    div#notebook-container    {{ width: {percent}%; }}
                    div#menubar-container     {{ width: {percent}%; }}
                    #header-container {{
                        width: {percent}%;
                            padding: 10px;
                    }}            
                </style>"""
        )
    )


def autostart():
    return display(
        HTML(
            """
            <script>
                require(
                    ['base/js/namespace', 'jquery'], 
                    function(jupyter, $) {
                        $(jupyter.events).on("kernel_ready.Kernel", function () {
                            jupyter.actions.call('jupyter-notebook:run-cell');
                            jupyter.actions.call('jupyter-notebook:save-notebook');
                        });
                    }
                );
            </script>"""
        )
    )


def fix_scroll_bars():
    style = """
            <style>
                body {
                    overflow: hidden;
                }
               .output_subarea pre {
                    overflow: hidden;
               }
            </style>
            """
    display(HTML(style))


def output_widget_not_scrollable():
    style = """
        <style>
           .jupyter-widgets-output-area .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
            .jupyter-widgets-output-area  {
                height: auto !important;
            }
        </style>
        """
    display(HTML(style))


def progress_bar_iterate(list_):
    list_ = list(list_)
    f = widgets.IntProgress(
        min=0, max=len(list_), step=1, description='Loading:', orientation='horizontal'
    )
    display(f)
    for item in list_:
        f.value += 1
        yield item
    f.close()


# noinspection PyTypeChecker
def hide_code(show: bool = False) -> None:
    display(
        HTML(
            """<script
  src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
  integrity="sha256-4+XzXVhsDmqanXGHaHvgh1gMQKX40OUvDEBTu8JcmNs="
  crossorigin="anonymous"></script>"""
        )
    )
    if show:
        display(HTML('<script >$(".jp-CodeCell").show()</script>'))
    else:
        display(HTML('<script >$(".jp-CodeCell").hide()</script>'))


def show_hide_code(default: bool = True) -> None:
    javascript_functions = {False: "hide()", True: "show()"}
    button_descriptions = {False: "Show code", True: "Hide code"}

    def toggle_code(code_state):
        """
        Toggles the JavaScript show()/hide() function on the div.input element.
        """

        output_string = "<script>$(\"div.input\").{}</script>"
        output_args = (javascript_functions[code_state],)
        output = output_string.format(*output_args)

        display(HTML(output))

    def button_action(value):
        """
        Calls the toggle_code function and updates the button description.
        """

        code_state = value.new

        toggle_code(code_state)

        value.owner.description = button_descriptions[code_state]

    state = default
    toggle_code(state)

    button = widgets.ToggleButton(state, description=button_descriptions[state])
    button.observe(button_action, "value")

    display(button)


def plot_dendrogram(axis, model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    dendrogram(linkage_matrix, ax=axis, **kwargs)


def figsize(width: float, height: float) -> None:
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = width
    fig_size[1] = height
    plt.rcParams["figure.figsize"] = fig_size


def plot_silhouette(X: DataFrame, y_pred: List[int], y_true: List[int], metric: str) -> None:
    y_pred = map_y_pred_by_prevalence(y_pred, y_true)
    y_pred = Series(y_pred)
    n_clusters = get_cluster_count(y_pred)
    sample_silhouette_values = silhouette_samples(X, y_pred, metric)
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(10, 5)
    y_lower = 0

    for i in range(n_clusters):
        cluster_index = y_pred == i
        labels = get_cluster_identifiers(y_true)
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_index]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        for label in labels:
            colormap = {0: 'tab:cyan', 1: 'tab:red'}
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                where=[y == label for y in y_true[cluster_index]],
                facecolor=colormap[label],
                edgecolor=colormap[label],
                alpha=1,
            )
            ax1.text(
                -0.05, y_lower + 0.7 * size_cluster_i, f'({chr(ord("`") + (i + 1))})', fontsize=12
            )
        y_lower = y_upper + 10

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    silhouette_avg = silhouette_score(X, y_pred, metric=metric)
    ax1.axvline(x=silhouette_avg, color="k", linestyle="--")
    ax1.set_yticks([])
    plot_style(axis=ax1)


def plot_nr_clusters_evaluation(X, range_n_clusters, protocol):
    points = []
    metric = "si"
    for n_clusters in range_n_clusters:
        y_pred = protocol.algorithm(X, n_clusters)
        score = protocol.measure_internal_metrics(X, y_pred)
        points.append(score[metric])
    plt.title("Silhouette index")
    plt.xticks(range_n_clusters)
    plt.xlabel("Number of clusters")
    plt.ylabel(metric)
    plt.bar(range_n_clusters, points)


def ascii_heatmap(data: Dict[str, Dict[str, float]], **kwargs) -> str:
    keys = list(data.keys())
    output = [["", *keys]]
    for key1 in keys:
        row = [key1]
        for key2 in keys:
            row.append(str(round(data[key1][key2], 3)))
        output.append(row)
    return tabulate_formatted(output, **kwargs)


def plot_heatmap(data: Dict[str, Dict[str, float]], ax=None, just_half: bool = False) -> None:
    ax = ax or plt.gca()
    x_labels = data.keys()
    y_labels = next(iter(data.values())).keys()

    data_matrix = [
        [
            data[y_label][x_label] if (not just_half or x_index <= y_index) else 0 \
            for x_index, x_label in enumerate(x_labels)
        ]
        for y_index, y_label in enumerate(y_labels)
    ]

    im = ax.imshow(data_matrix)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            value = data_matrix[i][j]
            ax.text(j, i, f'{value:.2f}' if value != 0 else "", ha="center", va="center", color="w")
    ax.figure.colorbar(im, ax=ax)


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


def display_statistics(
        protocol: ClusteringProtocol, X: DataFrame, y_true: Series, y_pred: List
) -> None:
    score = protocol.measure_metrics(X, y_pred, y_true)
    print(text_title("ClassificationMetrics"))
    print(dict_to_table_vertical(dict(score)))
    print()
    print(text_title("Real label distribution"))
    print(format_cluster_real_labels(get_counts_per_cluster(y_pred, y_true)))

    print()
    print(text_title("Feature statistic"))
    statistic = measure_cluster_features_statistics(pd.concat([X], axis=1), y_pred)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(statistic)


def plot_roc_from_result_vector(y: Series, result: ModelResult, label: str = None, plot_kwargs: Mapping = None) -> None:
    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}

    fpr, tpr, _ = roc_curve(y.loc[result['y_test_score'].index], result['y_test_score'])
    auc_value = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, lw=1, label=f'{"ROC curve" if not label else label} (AUC=%0.3f)' % auc_value, **plot_kwargs
    )
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_roc_from_result(y: Series, result: ModelCVResult, label: str = None, plot_kwargs: Mapping = None) -> None:
    plot_roc_from_result_vector(y, get_result_vector_from_result(result), label, plot_kwargs=plot_kwargs)


def plot_roc_from_results_averaged(
        y: Series, results: List[ModelCVResult], label: str = None
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
        normalized_fpr,
        mean_tpr,
        lw=0.1,
        label=f'{"ROC curve" if not label else label} (AUC=%0.3f ±%0.3f)' % (mean_auc, std_auc)
    )
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def plot_roc_from_results_threshold_averaged(
        y: Series, results: List[ModelCVResult], label: str = None
) -> None:
    lw = 2

    fpr, tpr, thresholds = compute_threshold_averaged_roc(y, results)

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


def notify(message: str = 'Done!') -> None:
    notify2.init('Python ML Development')
    n = notify2.Notification(message)
    n.set_timeout(notify2.EXPIRES_NEVER)
    n.set_urgency(notify2.URGENCY_CRITICAL)
    n.show()


def plot_feature_importance(coefficients: DataFrame, limit: int = None) -> None:
    with pd.option_context(
            'display.max_rows', None, 'display.max_columns', None
    ):
        if not limit:
            limit = len(coefficients)

        coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=True, by='mean').index)
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

    axis.title.set_position([.5, 1.03])
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
    plt.savefig(*args, **kwargs, bbox_inches='tight', pad_inches=0.1)


italic = {'font-style': 'italic'}

template = [
    Category(indent=0, label='Anthropometrics'),
    Attribute(indent=1, key='AGE', label='Age, y'),
    Attribute(indent=1, key='SEX', label='Female, n (%)'),
    Attribute(indent=1, key='BW', label='Body weight, kg'),
    Attribute(indent=1, key='BMI', label='Body mass index, kg/m²'),
    Attribute(indent=1, key='WAISTC', label='Waist circumference, cm'),
    Attribute(indent=1, key='WHR', label='Waist-hip ratio'),
    Attribute(indent=1, key='SKINF', label='Skinfold, cm'),
    Category(indent=0, label='Hemodynamics'),
    Attribute(indent=1, key='SBP', label='Systolic BP, mm Hg'),
    Attribute(indent=1, key='DBP', label='Diastolic BP, mm Hg'),
    Attribute(indent=1, key='PP', label='Pulse pressure, mm Hg'),
    Attribute(indent=1, key='MBP', label='MAP, mm Hg'),
    Attribute(indent=1, key='PR', label='Heart rate, bpm'),
    Category(indent=0, label='Questionnaire data'),
    Attribute(indent=1, key='SMK', label='Current or past smoking, n (%)'),
    Attribute(indent=1, key='ALC1', label='Drinking alcohol, n (%)'),
    Attribute(indent=1, key='COF', label='Caffeine-containing beverages, n (%)'),
    Attribute(indent=1, key='A_SPORT', label='Practice sports on a regular basis, n (%)'),
    Attribute(indent=1, key='WA_NOW', label='Walks on a regular basis, n (%)'),
    Attribute(indent=1, key='SS', label='Psychological tensions and stress, score'),
    Attribute(indent=1, key='SOCK', label='Social class, 0/1/2/3, %'),
    Category(indent=0, label='Drug treatment'),
    Attribute(indent=1, key='TRT_HT', label='Treated for hypertension, n (%)'),
    CategoryStyled(indent=2, label='Class of AHT', style=italic),
    Attribute(indent=2, key='TRT_BB', label='Beta blocking agents, n (%)'),
    Attribute(indent=2, key='TRT_CEB', label='Calcium entry blockers, n (%)'),
    Attribute(indent=2, key='TRT_ACE', label='ACE blockers, n (%)'),
    Attribute(indent=2, key='TRT_ARA', label='ARA blockers, n (%)'),
    Attribute(indent=2, key='TRT_DD', label='Diuretics, n (%)'),
    Attribute(indent=1, key='NSAID', label='Non-steroidal antiflogistic drugs, n (%)'),
    Attribute(indent=1, key='APDRG', label='Anti-platelet drugs, n (%)'),
    Category(indent=0, label='History of disease data'),
    Attribute(indent=1, key='HHT', label='Hypertensive, n (%)'),
    Attribute(indent=1, key='HDM', label='History of diabetes mellitus, n (%)'),
    Attribute(indent=1, key='HCAR2', label='History of cardiac disease, n (%)'),
    Attribute(indent=1, key='HCV2', label='History of cardiovascular disease, n (%)'),
    Category(indent=0, label='Biochemical data'),
    CategoryStyled(indent=1, label='Blood counts', style=italic),
    Attribute(indent=2, key='RBC', label='Red blood cell, 1012/L'),
    Attribute(indent=2, key='HTC', label='Haematocrit, %'),
    Attribute(indent=2, key='HGB', label='Haemoglobin, mmol/L'),
    Attribute(indent=2, key='MCV', label='Mean corpuscular volume, 10-15 L'),
    Attribute(indent=2, key='MCH', label='Mean corpuscular hemoglobin, fmol/cell'),
    Attribute(indent=2, key='LFERR', label='Serum ferritin, ng/mL'),  # TODO: LOG
    Attribute(indent=2, key='WBC', label='White blood cell, 109/L'),
    Attribute(indent=2, key='LYMF', label='Lymphocytes, %'),
    Attribute(indent=2, key='MONO', label='Monocytes, %'),
    Attribute(indent=1, key='LGGT', label='Gamma glutamyl transferase, mmol/l'),  # TODO: LOG
    Attribute(indent=1, key='BSUG', label='Blood sugar, mmol/l'),
    CategoryStyled(indent=1, label='Lipid profile', style=italic),
    Attribute(indent=2, key='TCHOL', label='Total cholesterol, mmol/L'),
    Attribute(indent=2, key='HCHOL', label='HDL cholesterol, mmol/L'),
    Attribute(indent=2, key='LCHOL', label='LDL cholesterol, mmol/L'),
    Attribute(indent=2, key='TRGL', label='Triglycerides, mmol/L'),
    CategoryStyled(indent=1, label='Hormones', style=italic),
    Attribute(indent=2, key='LPRA', label='Plasma renin activity, ng/L/sec'),  # TODO: log
    Attribute(indent=2, key='LINS', label='Insulin, μmol/L'),  # TODO: log
    Attribute(indent=2, key='LLEPT', label='Leptin, ng/mL'),  # TODO: log
    CategoryStyled(indent=1, label='Minerals', style=italic),
    Attribute(indent=2, key='SNA', label='Serum Na, mmom/L'),
    Attribute(indent=2, key='SK', label='Serum K, mmom/L'),
    CategoryStyled(indent=1, label='Nitrogenous waste in blood', style=italic),
    Attribute(indent=2, key='SCRT', label='Serum creatinine, µmol/L'),
    Attribute(indent=2, key='SUA', label='Serum uric acid, µmol/L'),
    CategoryStyled(indent=1, label='Urine measurements (excretion)', style=italic),
    Attribute(indent=2, key='NA', label='Na, mmol/24h'),
    Attribute(indent=2, key='K', label='K, mmol/24h'),
    Attribute(indent=2, key='LALDO', label='Aldosterone, nmol/24h'),  # TODO: log
    Attribute(indent=2, key='LCRTSL', label='Cortisol, nmol/24h'),  # TODO: log
    Category(indent=0, label='ECG'),
    CategoryStyled(indent=1, label='Duration', style=italic),
    Attribute(indent=2, key='PQD', label='PQ interval, ms'),
    Attribute(indent=2, key='QSD', label='QRS interval, ms'),
    Attribute(indent=2, key='QTCD', label='QT interval, corrected, ms'),
    CategoryStyled(indent=1, label='Amplitude ', style=italic),
    Attribute(indent=2, key='SA_V3', label='S wave in V3, mm'),
    Attribute(indent=2, key='RA1_AVL', label='R wave in aVL, mm'),
    Attribute(indent=2, key='RA1_V5', label='R wave in V5, mm'),
    Attribute(indent=2, key='PA_AVG', label='P wave, leads I,II and aVF, mm'),
    Attribute(indent=2, key='TA_AVG', label='T wave (leads I, II, V3, V4, V5, V6, aVL, aVF), mm'),
    CategoryStyled(indent=1, label='Products ', style=italic),
    Attribute(indent=2, key='CORNELL_PROD', label='Cornell'),
    Attribute(indent=2, key='SOKOLOW_LYON', label='Sokolow-Lyon'),
]


# format_item


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


def feature_statistics_to_html_table(statistics: DataFrame) -> str:
    return data_frame_to_html_table(statistics, 'Characteristic (feature)')


def data_frame_to_html_table(data_frame: DataFrame, index_label: str = None) -> str:
    html = """
        <style>
            .category {
                font-weight: bold;
            }
        </style>
    """

    html += '<table>'
    html += f'<thead><th>{index_label}</th>{"".join(("<th>" + title + "</th>" for title in data_frame.columns))}</thead>'
    for item in template:
        html += '<tr>'
        html += format_item(item, data_frame)
        html += '</tr>'

    return html


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


def qgrid(*args, **kwargs):
    import qgrid
    return qgrid.show_grid(*args, **kwargs, grid_options={'forceFitColumns': False, 'defaultColumnWidth': 200},
                           show_toolbar=True)


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
