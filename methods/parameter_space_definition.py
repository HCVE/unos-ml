import numpy
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, CondensedNearestNeighbour, \
    EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, \
    NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks
from sklearn.impute import SimpleImputer
from sklearn.impute._iterative import IterativeImputer

from methods.parameter_space import Parameter, DiscreteOptionDomain, Class, select_control, parameter, ParameterGroup, \
    DiscreteUniformDomain, ContinuousUniformDomain
from methods.transformers import AutoSMOTENC


def reduce_classes_parameter():
    return Parameter('reduce_classes', DiscreteOptionDomain([True], default=True))


def imputer_parameter(name):
    return Parameter(
        name,
        DiscreteOptionDomain(
            [
                # Class(IterativeImputer),
                Class(SimpleImputer),
            ],
            ui_factory=select_control(switchable=True),
            default=Class(SimpleImputer)
        ),
        ui_factory=parameter("Imputer"),
    )


def resample_parameter():
    return Parameter(
        "upsampler",
        DiscreteOptionDomain(
            [
                None,
                Class(RandomOverSampler, parameters={'random_state': 456}),
                Class(RandomUnderSampler),
                Class(
                    AutoSMOTENC,
                    # domain=ParameterGroup(
                    #     [
                    #         # TODO Multiclass: Parameter('sampling_strategy', ContinuousUniformDomain(0.20, 1)),
                    #         Parameter('k_neighbors', DiscreteUniformDomain(1, 6)),
                    #     ]
                    # )
                ),
                Class(
                    ClusterCentroids,
                    # domain=ParameterGroup(
                    #     [
                    #         # TODO Multiclass: Parameter('sampling_strategy', ContinuousUniformDomain(0.20, 1)),
                    #     ]
                    # )
                ),
                Class(
                    CondensedNearestNeighbour,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         # TODO disabled: bad naming convention
                    #         # Parameter('n_seeds_S', DiscreteUniformDomain(1, 6)),
                    #     ]
                    # )
                ),
                Class(
                    EditedNearestNeighbours,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         Parameter('kind_sel', DiscreteOptionDomain(['all', 'mode'])),
                    #     ]
                    # )
                ),
                Class(
                    ADASYN,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         Parameter('kind_sel', DiscreteOptionDomain(['all', 'mode'])),
                    #     ]
                    # )
                ),
                Class(
                    RepeatedEditedNearestNeighbours,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         Parameter('max_iter', DiscreteUniformDomain(50, 200)),
                    #         Parameter('kind_sel', DiscreteOptionDomain(['all', 'mode'])),
                    #     ]
                    # )
                ),
                Class(
                    AllKNN,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         Parameter('kind_sel', DiscreteOptionDomain(['all', 'mode'])),
                    #     ]
                    # )
                ),
                Class(
                    InstanceHardnessThreshold,
                    # domain=ParameterGroup(
                    #     [
                    #         # TODO Multiclass: Parameter('sampling_strategy', ContinuousUniformDomain(0.20, 1)),
                    #     ]
                    # )
                ),
                Class(
                    NearMiss,
                    # domain=ParameterGroup(
                    #     [
                    #         # TODO Multiclass: Parameter('sampling_strategy', ContinuousUniformDomain(0.20, 1)),
                    #         Parameter('version', DiscreteOptionDomain([1, 2, 3])),
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         # TODO: omitted
                    #         # n_neighbors_ver3int
                    #     ]
                    # )
                ),
                Class(
                    NeighbourhoodCleaningRule,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         Parameter('threshold_cleaning', ContinuousUniformDomain(0.1, 0.9)),
                    #     ]
                    # )
                ),
                Class(
                    OneSidedSelection,
                    # domain=ParameterGroup(
                    #     [
                    #         Parameter('n_neighbors', DiscreteUniformDomain(1, 6)),
                    #         # TODO disabled: bad naming convention
                    #         # Parameter('n_seeds_S', DiscreteUniformDomain(1, 6)),
                    #     ]
                    # )
                ),
                Class(TomekLinks),
            ],
            ui_factory=select_control(switchable=True),
            default=Class(RandomOverSampler)
        ),
        ui_factory=parameter("Upsampler"),
    )


def learning_rate_parameter():
    return Parameter(
        "learning_rate", DiscreteOptionDomain((10.0**numpy.arange(-2, 1)).tolist() + [0.2, 0.3])
    )
