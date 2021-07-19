import random
from collections import defaultdict
from functools import singledispatch
from itertools import product
from typing import Dict, Any, Union, Iterable, TypeVar, List

import numpy
from attr import dataclass
from functional_pipeline import pipeline
from pyramda import find
from sklearn.impute import SimpleImputer
from toolz.curried import get

from api_utils import get_member_by_path
from functional import does_objects_equal
from utils import qualified_name

imputers = [
    SimpleImputer(),
]

T = TypeVar('T')


class Displayable:

    def __init__(self, ui_factory):
        try:
            self.control = ui_factory(self)
        except TypeError:
            pass


class ParameterGroup(Displayable):

    def __init__(self, parameters, ui_factory=None, name=None, value=None):
        self.parameters = parameters
        self.name = name
        self.value = value or {'type': 'dict'}
        super().__init__(ui_factory or parameter_group())


class Class(Displayable):

    def __init__(self, name, parameters=None, domain=None, ui_factory=None):
        if type(name) == str:
            self.name = name
        else:
            self.name = qualified_name(name)
        self.parameters = parameters
        self.domain = domain
        super().__init__(ui_factory or class_control())


@dataclass
class ClassInstance:
    name: str
    parameters: Dict[str, Any]


class Parameter(Displayable):

    def __init__(self, name, domain, ui_factory=None, default=None):
        self.name = name
        self.domain = domain
        self.default = default
        super().__init__(ui_factory or parameter(name))


class ContinuousUniformDomain(Displayable):

    def __init__(
        self,
        start,
        end,
        num=20,
        ui_factory=None,
        default=None,
    ):
        self.end = end
        self.start = start
        self.num = num
        self.default = default or start
        super().__init__(ui_factory or range_control((self.end - self.start) / num))

    @property
    def values(self):
        return numpy.linspace(self.start, self.end, num=self.num).tolist()


class DiscreteOptionDomain(Displayable):

    def __init__(self, options, ui_factory=None, default=None):
        self.options = options
        try:
            self.default = find(default, self.options)
        except TypeError:
            self.default = default or options[0]
        super().__init__(ui_factory or select_control())

    @property
    def values(self):
        return self.options


class DiscreteUniformDomain(Displayable):

    def __init__(self, start, end, step=1, ui_factory=None, default=None):
        self.end = end
        self.start = start
        self.step = step
        self.default = default or start
        super().__init__(ui_factory or range_control(step))

    @property
    def values(self):
        return (numpy.arange(self.start, self.end, step=self.step)).tolist()


def range_control(step=1):
    return lambda param: {
        'type': 'Range',
        'step': step,
    }


def select_control(switchable=False):
    return lambda domain: {
        'type': 'Select',
        'domain': domain.options,
        'switchable': switchable,
    }


def text_input(width=50):
    return lambda _: {
        'type': 'TextInput',
        'width': width,
    }


def from_to_input():
    return lambda domain: {
        'type': 'FromToInput',
    }


def parameter(label):
    return lambda _: {
        'type': 'Parameter',
        'label': label,
    }


def parameter_group():
    return lambda group: {
        'type': 'ParameterGroup',
    }


def class_control():
    return lambda _: {
        'type': 'Class',
    }


def render_domain(label):
    return lambda param: {
        'type': 'RenderDomain',
        'label': label,
        'domain': param.domain,
    }


@singledispatch
def generate_individual(value):
    return value


@generate_individual.register(DiscreteUniformDomain)
def _(domain):
    try:
        return random.randrange(domain.start, domain.end, domain.step)
    except AttributeError:
        return None


@generate_individual.register(ParameterGroup)  # type: ignore
def _(group):
    return {item.name: generate_individual(item) for item in group.parameters}


@generate_individual.register(Parameter)  # type: ignore
def _(param):
    return generate_individual(param.domain)


@generate_individual.register(DiscreteOptionDomain)  # type: ignore
def _(domain):
    return generate_individual(random.choice(domain.options))


@generate_individual.register(ContinuousUniformDomain)  # type: ignore
def _(domain):
    return generate_individual(random.uniform(domain.start, domain.end))


@generate_individual.register(Class)  # type: ignore
def _(cls):
    return ClassInstance(
        name=cls.name,
        parameters={
            **(cls.parameters or {}),
            **generate_individual(cls.domain or {})
        }
    )


@generate_individual.register(DiscreteUniformDomain)  # type: ignore
def _(domain):
    try:
        return random.choice(range(domain.start, domain.end, domain.step))
    except AttributeError:
        return None


Configuration = Dict[str, Union[ClassInstance, int, str, bool]]

InstanceConfiguration = Dict[str, Union[object, int, str, bool]]


@singledispatch
def generate_configuration_series(value: T) -> Iterable[T]:
    return [value]


@generate_configuration_series.register  # type: ignore
def _(group: ParameterGroup) -> Iterable[Configuration]:
    keys = [p.name for p in group.parameters]
    parameter_values = (generate_configuration_series(param) for param in group.parameters)
    parameters_combined = (product(*parameter_values))
    yield from (dict(zip(keys, parameters)) for parameters in parameters_combined)


@generate_configuration_series.register  # type: ignore
def _(param: Parameter) -> Iterable[Configuration]:
    return generate_configuration_series(param.domain)


@generate_configuration_series.register(DiscreteUniformDomain)  # type: ignore
@generate_configuration_series.register(ContinuousUniformDomain)
@generate_configuration_series.register(DiscreteOptionDomain)
def _(domain) -> Iterable[Configuration]:
    for value in domain.values:
        yield from generate_configuration_series(value)


@generate_configuration_series.register  # mypy: ignore
def _(cls: Class) -> Iterable[ClassInstance]:
    yield from (
        ClassInstance(
            name=cls.name, parameters={
                **(cls.parameters or {}),
                **generated_parameters,
            }
        ) for generated_parameters in generate_configuration_series(cls.domain or {})
    )


# get_defaults


@singledispatch
def get_defaults(value):
    try:
        return get_defaults(value.default)
    except AttributeError:
        return value


@get_defaults.register(ParameterGroup)  # type: ignore
def _(group):
    return {item.name: get_defaults(item) for item in group.parameters}


@get_defaults.register(Parameter)  # type: ignore
def _(param):
    return get_defaults(param.domain)


@get_defaults.register(Class)  # type: ignore
def _(cls):
    try:
        return ClassInstance(
            name=cls.name,
            parameters={
                **(cls.parameters or {}),
                **(get_defaults(cls.domain) or {}),
            }
        )
    except AttributeError:
        return None


@singledispatch
def instantiate_configuration(value: T) -> T:
    return value


# noinspection Mypy
@instantiate_configuration.register(dict)
def _(d: dict) -> InstanceConfiguration:
    return {key: instantiate_configuration(value) for key, value in d.items()}


# noinspection Mypy
@instantiate_configuration.register(ClassInstance)
def _(d: ClassInstance) -> object:
    Cls = get_member_by_path(d.name)
    return Cls(**(d.parameters or {}))


def configuration_to_params(dictionary: Dict) -> Dict:
    return_value = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                return_value["%s__%s" % (key, key2)] = value2
        else:
            return_value[key] = value

    return return_value


def dict_configuration_to_params(configuration):
    parameters = pipeline(
        configuration,
        [
            instantiate_configuration,
            get('pipeline'),
            configuration_to_params,
        ],
    )
    return parameters


def unique_list(list1):
    output = []
    for x in list1:
        if not any((does_objects_equal(x, y) for y in output)):
            output.append(x)
    return output


def configurations_to_grid_search(
    configurations: Iterable[Union[InstanceConfiguration, Configuration]]
) -> Dict[str, List[Any]]:
    grid_params = defaultdict(list)
    for configuration in configurations:
        for key, value in configuration.items():
            grid_params[key].append(value)
    grid_params_unique = {key: unique_list(value) for key, value in grid_params.items()}
    return grid_params_unique
