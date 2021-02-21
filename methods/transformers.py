from imblearn.over_sampling import SMOTENC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from methods.methods_utils import get_categorical_features


class OneHotCategoriesAuto(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.onehots = {}
        self.columns = []

    def transform(self, X):
        X_new = X.copy()
        X_new.drop(self.columns, inplace=True, axis=1)
        for categorical_column in self.columns:
            encoded = self.onehots[categorical_column].transform(X[[categorical_column]])
            for index, new_column in enumerate(encoded.T):
                X_new["%s__%s" %
                      (categorical_column, self.onehots[categorical_column].categories_[0][index])
                      ] = new_column
        return X_new

    def fit(self, X, y=None, **fit_params):
        self.columns = []
        self.onehots = {}
        for column in X.columns:
            if (X[column].nunique() <= 5):
                self.columns.append(column)

        for categorical_column in self.columns:
            self.onehots[categorical_column] = OneHotEncoder(sparse=False, categories='auto')
            self.onehots[categorical_column].fit(X[[categorical_column]])

        return self


class AutoSMOTENC(SMOTENC):
    ...

    def __init__(self, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=1):
        super().__init__(
            categorical_features=[],
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def _fit_resample(self, X, y):
        self.categorical_features = get_categorical_features(X)
        return super()._fit_resample(X, y)
