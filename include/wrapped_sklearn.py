from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from include.utils import DFWrapped


class DFOneHotEncoder(DFWrapped, OneHotEncoder):
    ...


class DFColumnTransformer(DFWrapped, ColumnTransformer):
    ...


class DFSimpleImputer(DFWrapped, SimpleImputer):
    ...


class DFStandardScaler(DFWrapped, StandardScaler):
    ...


class DFOrdinalEncoder(DFWrapped, OrdinalEncoder):
    ...


class DFKNNImputer(DFWrapped, KNNImputer):
    ...


class DFPipeline(Pipeline):

    def get_feature_names(self):
        return self.steps[-1][1].fitted_feature_names


class DFLogisticRegression(DFWrapped, LogisticRegression):
    ...


class DFXGBClassifier(DFWrapped, XGBClassifier):
    ...
