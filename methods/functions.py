import pickle
from dataclasses import dataclass

from pandas import Series, DataFrame


def lvddf_to_1_class_value(value):
    return 0 if (value == 0 or value == 1) else 1


def lvddf_to_1_class(series):
    return series.map(lvddf_to_1_class_value)


def lvddf_to_1_class_proba(data_frame: DataFrame) -> DataFrame:
    return DataFrame(
        {
            'y_predict_probabilities_0': data_frame.iloc[:, 0] + data_frame.iloc[:, 1],
            'y_predict_probabilities_1': data_frame.iloc[:, 2]
        }
    )


def par_right(n, string):
    return string + (n - len(string)) * "0" if n > len(string) else string


def format_ratio(number):
    rounded = round(number, 3)
    if 0 <= number < 1:
        return par_right(4, str(rounded)[1:])
    else:
        return str(rounded)


def get_optimized_parameters(type, metric):
    with open("./output/%s/random_search" % type, "rb") as f:
        list = pickle.load(f)
        list.sort(key=lambda i: i['metrics'][metric], reverse=True)
        try:
            return list[0]['parameters']
        except IndexError:
            raise Exception("Optimized parameters not present")


@dataclass
class EvaluationResult:
    y: Series
    y_predict: Series
    y_score: Series


if __name__ == '__main__':
    print(get_optimized_parameters("adaboost", "f1"))
