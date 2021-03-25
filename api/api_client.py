import asyncio
from dataclasses import dataclass
from typing import Dict, Any

import websockets
from aioprocessing import AioQueue
from inflection import camelize
from pyramda import curry

from api.api_utils import json_serialize, json_serialize_types
from utils import object2dict
from custom_types import ClassificationMetrics


class WebsocketQueue:
    queue: AioQueue

    def __init__(self):
        self.queue = AioQueue()

    async def send(self, message):
        self.queue.put(message)


class ApiClient:

    async def connect(self):
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            self.websocket = websocket
            await websocket.recv()

    async def send(self, message):
        await self.websocket.send(message)
        await self.websocket.recv()

    async def close(self):
        await self.websocket.close()


def api_client_factory():
    client = ApiClient()
    asyncio.get_event_loop().run_until_complete(client.connect())
    return client


@curry
async def api_send(message, websocket=None):
    serialized_message = json_serialize(camelize_recursive(json_serialize_types(message)))
    if websocket:
        await websocket.send(serialized_message)
    else:
        uri = "ws://localhost:8765"
        async with websockets.connect(uri, max_size=2 ** 25) as websocket:
            await websocket.send(serialized_message)


def await_api_send(message, websocket=None):
    asyncio.get_event_loop().run_until_complete(api_send(message, websocket=websocket))


def output_ready(data):
    return event('OUTPUT_READY', data)


def record(record_type: str, data: Any, key: str = None) -> Dict:
    return {'type': record_type, 'data': data, 'key': key or record_type}


def external_event(data):
    return event('EXTERNAL_EVENT_GENERATED', data)


def running_optimization_list_ready(running_list):
    return event('RUNNING_OPTIMIZATION_LIST_READY', {
        'list': running_list,
    })


@dataclass
class TrainingCurvePoint:
    n_samples: int
    metrics_test: ClassificationMetrics
    metrics_train: ClassificationMetrics


def training_curve_point_ready(point: TrainingCurvePoint, key: str, finished: bool = False) -> dict:
    return event(
        'TRAINING_CURVE_POINT_READY',
        object2dict({
            'point': point,
            'key': key,
            'finished': finished,
        })
    )


def event(name, payload):
    return {
        'type': name,
        'payload': payload,
    }


def optimization_configuration_ready(
        results, records, parameters, model_type, features, label, elapsed
):
    return {
        'type': 'OPTIMIZATION_CONFIGURATION_READY',
        'payload': {
            'results': results,
            'records': records,
            'parameters': parameters,
            'model_type': model_type,
            'features': features,
            'label': label,
            'elapsed': elapsed,
        },
    }


def camelize_adjusted(string: str) -> str:
    if len(string) == 1:
        return string
    else:
        return camelize(string, False)


def camelize_recursive(d):
    if isinstance(d, dict):
        new = {}
        for k, v in d.items():
            new[camelize_adjusted(k) if k[0] != "_" else k] = camelize_recursive(v)
        return new
    elif isinstance(d, list):
        return list(map(camelize_recursive, d))
    else:
        return d


if __name__ == '__main__':
    pass
