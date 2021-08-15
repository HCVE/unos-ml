from functools import reduce
from typing import Callable, TypeVar, Iterable, List, Any, Union, Sequence, Dict

import numpy as np

from include.custom_types import IndexAccess

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def t(*args):
    print(*args)
    return args[-1]


def unpack_args(function: Callable[..., T1]) -> Callable[[Iterable], T1]:

    def unpacked(args):
        return function(*args)

    return unpacked


def or_fn(*fns: Callable[..., bool]) -> Callable[..., bool]:
    return lambda *args: reduce(lambda current_value, fn: current_value or fn(*args), fns, False)


def compact(iterable: Iterable) -> Iterable:
    return filter(lambda i: i is not None, iterable)


def try_except(try_clause: Callable, except_clauses: Dict) -> Any:
    # noinspection PyBroadException
    try:
        return try_clause()
    # noinspection PyBroadException
    except Exception as e:
        for ExceptionClass, except_clause in except_clauses.items():
            if isinstance(e, ExceptionClass):
                return except_clause()
        raise e


def mapl(func, iterable):
    return list(map(func, iterable))


def flatten(iterable_outer: Iterable[Union[Iterable[T1], T1]]) -> Iterable[T1]:
    for iterable_inner in iterable_outer:
        if isinstance(iterable_inner, Iterable) and (not isinstance(iterable_inner, str)):
            for item in iterable_inner:
                yield item
        else:
            yield iterable_inner


def flatten_recursive(list_of_lists: List) -> List:
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], Sequence) or isinstance(list_of_lists[0], np.ndarray):
        return [*flatten_recursive(list_of_lists[0]), *flatten_recursive(list_of_lists[1:])]
    return [*list_of_lists[:1], *flatten_recursive(list_of_lists[1:])]


def find(callback: Callable[[T1], bool], list_to_search: Iterable[T1]) -> T1:
    return next(filter(callback, list_to_search))


def find_index(
    callback: Callable[[T1], bool], list_to_search: Union[List[T1], str], reverse=False
) -> int:
    if reverse:
        iterable = add_index_reversed(list_to_search)
    else:
        iterable = add_index(list_to_search)
    return next(filter(lambda item: callback(item[1]), iterable))[0]


def add_index(iterable: Iterable) -> Iterable:
    for (index, item) in enumerate(iterable):
        yield index, item


def add_index_reversed(iterable: Union[List, str]) -> Iterable:
    for index in reversed(range(len(iterable))):
        yield index, iterable[index]


def pipe(*args: Any) -> Any:
    current_value = args[0]
    for function in args[1:]:
        current_value = function(current_value)
    return current_value


def pass_args(define, to):
    return to(*define)


def statements(*args: Any) -> Any:
    return args[-1]


def unzip(iterable: Iterable) -> Iterable:
    return zip(*iterable)


TIndexAccess = TypeVar('TIndexAccess', bound=IndexAccess)


def tap(callback: Callable[[T1], None]) -> Callable[[T1], T1]:

    def tap_callback(arg: T1) -> T1:
        callback(arg)
        return arg

    return tap_callback
