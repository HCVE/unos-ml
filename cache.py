import os
import pickle

from joblib import Memory
from portalocker import Lock

memory = Memory('./temporary/cache_1', verbose=2)


def get_memory(identifier: str, verbose: int = 0) -> Memory:
    return Memory(f'./temporary/cache/{identifier}', verbose=verbose)


def get_file_name(identifier: str) -> str:
    return identifier


def save_generic(folder, identifier, data):
    with open(f'./{folder}/%s' % get_file_name(identifier), "wb") as file:
        pickle.dump(data, file)


def load_generic(folder, identifier):
    with open(f'./{folder}/%s' % get_file_name(identifier), "rb") as file:
        return pickle.load(file)


def save_data(identifier, data):
    save_generic('data', identifier, data)


def load_data(identifier):
    return load_generic('data', identifier)


def save_cache(identifier, data):
    save_generic('cache', identifier, data)


def load_cache(identifier):
    return load_generic('cache', identifier)


def append_data(identifier, data):
    identifier = list(map(str, identifier))
    directory = "data/" + "/".join(identifier[:-1])
    filename = identifier[-1] + ".pickle"
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    path = "%s/%s" % (directory, filename)

    try:
        with open(path, "rb") as f:
            file_content = pickle.load(f)
    except (FileNotFoundError, EOFError):
        file_content = []

    with Lock(path, "wb") as f:
        file_content.append(data)
        pickle.dump(file_content, f)
