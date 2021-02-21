import asyncio

import numpy
from hyperopt.pyll import scope
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from toolz import merge

from api.api_client import output_ready, api_send, external_event
from utils import get_features_from_config, load_global_config
from methods.methods_utils import run_training_curve_generic, optimize_generic
from custom_types import MethodInfo, Method
from methods.running import run_generic
from methods.parameter_space import ParameterGroup, Parameter, DiscreteOptionDomain, Class, \
    select_control, parameter, DiscreteUniformDomain, ContinuousUniformDomain, configuration_to_params, get_defaults
from methods.parameter_space_definition import reduce_classes_parameter, imputer_parameter, resample_parameter
from hyperopt import hp


async def run(
        *args,
        **kwarg,
):
    await run_generic(
        RandomForestMethod.get_pipeline, get_parameter_space=get_parameter_space, *args, **kwarg
    )


async def run_training_curve(
        *args,
        **kwargs,
):
    await run_training_curve_generic(RandomForestMethod.get_pipeline, *args, **kwargs)


async def optimize(*args, **kwargs):
    await optimize_generic(RandomForestMethod.get_pipeline, get_parameter_space, *args, **kwargs)


def get_parameter_space():
    return ParameterGroup(
        [
            reduce_classes_parameter(),
            ParameterGroup(
                name="pipeline",
                parameters=[
                    imputer_parameter('imputer'),
                    # resample_parameter(),
                    Parameter(
                        name='classifier',
                        domain=ParameterGroup(
                            [
                                Parameter(
                                    "n_estimators",
                                    DiscreteUniformDomain(5, 505, step=100, default=200)
                                ),
                                # Parameter("criterion", DiscreteOptionDomain(['gini', 'entropy'])),
                                Parameter("max_depth", DiscreteOptionDomain([None, 1, 3])),
                                Parameter(
                                    "min_samples_split", DiscreteUniformDomain(2, 152, step=50)
                                ),
                                Parameter(
                                    "min_samples_leaf", DiscreteUniformDomain(1, 61, step=20)
                                ),
                                # Parameter(
                                #     "min_weight_fraction_leaf", ContinuousUniformDomain(0.0, 0.5)
                                # ),
                                Parameter("max_features", DiscreteOptionDomain(['sqrt', None])),
                                # Parameter(
                                #     "max_leaf_nodes", DiscreteOptionDomain([None, *range(2, 10)])
                                # ),
                                # Parameter("oob_score", DiscreteOptionDomain([False, True])),
                                # TODO: min_impurity_decrease
                                # Parameter("class_weight", DiscreteOptionDomain(['balanced'])),
                            ],
                        ),
                        ui_factory=parameter("Classifier"),
                    )
                ]
            )
        ]
    )


class RandomForestMethod(Method):

    @staticmethod
    def get_hyperopt_space():
        return hp.choice(
            'base', [
                {'classifier': {
                    'bootstrap': hp.choice('classifier_bootstrap', [True, False]),
                    'max_depth': scope.int(hp.quniform('max_depth', 2, 16, 1)),
                    'max_features': hp.choice('classifier_max_features', ['auto', 'log2', None]),
                    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
                    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
                    'n_estimators': scope.int(hp.uniform('classifier__n_estimators', 5, 500)),
                }}
            ]
        )

    @staticmethod
    def get_pipeline():
        classifier = Pipeline(
            [
                ('onehot', None),
                ('imputer', None),
                ('upsampler', None),
                ("classifier", RandomForestClassifier(n_jobs=1)),
            ],
        )
        return classifier

    def get_method():
        return RandomForestMethod

    if __name__ == '__main__':
        asyncio.get_event_loop().run_until_complete(
            run(
                output=lambda message: api_send(external_event(output_ready(message))),
                pipeline=configuration_to_params(get_defaults(get_parameter_space())),
                features=get_features_from_config(load_global_config()),
            )
        )
