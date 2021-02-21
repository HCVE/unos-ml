from importlib import import_module
from operator import itemgetter

from functional_pipeline import pipeline as pipeline
from numpy import mean
from pandas import Series, DataFrame
from toolz.curried import map, partial

from evaluation_functions import ModelCVResult, compute_curves, get_1_class_y_score
from formatting import format_feature, format_feature_detailed
from methods.functions import format_ratio


def get_module_by_path(path):
    return import_module(path)


def get_member_by_path(path):
    path_list = path.split(".")
    module = import_module(".".join(path_list[:-1]))
    return getattr(module, path_list[-1])


async def api_call_function(path, output, parameters):
    await get_member_by_path(path)(output=output, **parameters)


def data_frame_to_correlation_record(data_frame):
    return {
        'ind': [str(n + 1) for n in range(len(data_frame))],
        'vars': [format_feature(feature) for feature in data_frame.columns.values.tolist()],
        'corr': data_frame.corr().values.tolist(),
        'dat': [data_frame[feature].tolist() for feature in data_frame.columns],
    }


def data_frame_to_data_table(data_frame, widths=None):
    return {
        'rowHeaders': [format_feature_detailed(row) for row in data_frame.index.values.tolist()],
        'columnHeaders': [
            format_feature_detailed(column) for column in data_frame.columns.values.tolist()
        ],
        'rowData': data_frame.values.tolist(),
        'widths': widths or [],
    }


def structure_cross_validation(cross_validation):
    return {
        'mean_fit_time': mean(cross_validation.get("fit_time", [])),
        'total_fit_time': sum(cross_validation.get("fit_time", [])),
        'mean_accuracy': mean(cross_validation.get("test_accuracy", [])),
        'average_precision': mean(cross_validation.get("test_average_precision", [])),
        'mean_roc_auc': mean(cross_validation.get("test_roc_auc", [])),
        'mean_recall': mean(cross_validation.get("test_recall", [])),
    }


def metrics_to_table(structure):
    return {
        'first_header': True,
        'data': [
            [
                # "μ fit time [s]",
                "Accuracy",
                "Recall",
                "Precision",
                "ROC/AUC",
            ],
            [
                # round(structure['execution_time'], 2),
                round(structure['accuracy'], 2),
                round(structure['recall'], 2),
                round(structure['precision'], 2),
                round(structure['roc_auc'], 2),
            ],
        ],
    }


def output_feature_importance(result: ModelCVResult):
    try:

        return {
            'label': "Feature importance",
            'type': 'featureImportance',
            'key': 'correlation_table',
            'data': {
                "features": list(
                    pipeline(
                        enumerate(result['feature_importance']),
                        [
                            map(
                                lambda item: {
                                    "name": result['feature_importance'].index[item[0]],
                                    "name_detailed": format_feature_detailed(
                                        result['feature_importance'].index[item[0]]
                                    ),
                                    "importance": item[1]
                                }
                            ),
                            list,
                            partial(sorted, key=itemgetter("importance"), reverse=True),
                        ],
                    )
                ),
            },
            'width': 700,
        }
    except (ValueError, TypeError):
        pass


def output_curves_folds(results: ModelCVResult, y: Series):
    return {
        'label': "Recall precision folds",
        'type': 'PlotRecord',
        'key': 'recal_precision_folds',
        'data': {
            'labels': {
                'x': 'Recall',
                'y': 'Precision',
            },
            'lines': pipeline(
                results['y_scores'], [
                    map(lambda y_score: compute_curves(get_1_class_y_score(y_score), y)),
                    enumerate,
                    map(
                        lambda index__result: {
                            'x': index__result[1].curve_horizontal,
                            'y': index__result[1].curve_vertical_recall_precision,
                            'label': index__result[0],
                        }
                    ),
                    list,
                ]
            ),
            'plotProps': {
                'yDomain': [0, 1],
                'width': 450,
                'height': 375
            },
        },
        'width': 700,
    }


def output_feature_importance_table(result: ModelCVResult):
    try:
        feature_importance = DataFrame({
            'importance': result['feature_importance']
        }).applymap(format_ratio)
        return {
            'label': "Feature importance",
            'type': 'dataTable',
            'key': 'correlation_table',
            'data': data_frame_to_data_table(feature_importance),
            'width': 700,
        }
    except ValueError:
        pass
