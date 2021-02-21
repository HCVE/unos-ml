import asyncio
import json
from functools import singledispatch, partial
from typing import List

import numpy
import pandas
from functional_pipeline import pipeline
from pandas import Series
from toolz.curried import keyfilter, valmap

from api.api_functions import get_member_by_path
from utils import make_instance
from custom_types import FeatureImportanceItem
from functional import decorate_unpack, pipe, mapl


def json_serialize(obj):
    repl = json_serialize_replace(obj)
    return json.dumps(repl)


def json_serialize_replace(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [json_serialize_replace(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: json_serialize_replace(v) for k, v in obj.items()}
    elif isinstance(obj, range):
        return list(obj)
    else:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating) and not numpy.isnan(obj):
            return float(obj)
        else:
            try:
                if numpy.isnan(float(obj)):
                    return None
                else:
                    return obj
            except (ValueError, TypeError):
                return obj


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


@singledispatch
def json_serialize_types(value):
    try:
        value.__dict__
    except AttributeError:
        return value

    return {
        '__type': type(value).__module__ + "." + type(value).__name__,
        **pipeline(
            value.__dict__,
            [
                keyfilter(lambda l: not l.startswith("_")),
                json_serialize_types,
            ],
        ),
    }


@json_serialize_types.register(dict)
def _(value):
    return valmap(json_serialize_types, value)


@json_serialize_types.register(tuple)  # type: ignore[no-redef]
@json_serialize_types.register(list)
def _(value):
    return list(map(json_serialize_types, value))


@json_serialize_types.register(int)  # type: ignore[no-redef]
@json_serialize_types.register(float)
@json_serialize_types.register(bool)
@json_serialize_types.register(str)
@json_serialize_types.register(type(None))
def _(value):
    return value


@singledispatch
def json_deserialize_types(value):
    return value


@json_deserialize_types.register(dict)  # type: ignore[no-redef]
def _(dictionary):
    try:
        parameters = pipeline(
            dictionary,
            [
                keyfilter(lambda k: k != '__type'),
                valmap(json_deserialize_types),
            ],
        )
        Class = get_member_by_path(dictionary['__type'])
        instance = make_instance(Class)
        for key, value in parameters.items():
            setattr(instance, key, json_deserialize_types(value))
        return instance
    except KeyError:
        return valmap(json_deserialize_types, dictionary)


async def merge(*iterables):
    iter_next = {it.__aiter__(): None for it in iterables}
    while iter_next:
        for it, it_next in iter_next.items():
            if it_next is None:
                fut = asyncio.ensure_future(it.__anext__())
                fut._orig_iter = it
                iter_next[it] = fut
        done, _ = await asyncio.wait(iter_next.values(), return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            # noinspection PyProtectedMember
            iter_next[fut._orig_iter] = None
            try:
                ret = fut.result()
            except StopAsyncIteration:
                # noinspection PyProtectedMember
                del iter_next[fut._orig_iter]
                continue
            yield ret


def structure_feature_importance(series: Series) -> List[FeatureImportanceItem]:
    return pipe(
        series,
        pandas.Series.items,
        partial(
            mapl,
            decorate_unpack(
                lambda feature, importance: {
                    'feature': feature,
                    'importance': importance
                }
            )
        ),
    )
