from dataclasses import dataclass
from typing import Any, Callable
import json

from flask import current_app
import flask

from .shared import ENGINES, cache

__all__ = ('call_next_available_pipeline', 'postfork', 'postfork_chain')

postfork_chain = []


@dataclass
class CUDAInfo:
    cuda_device_count = 0


cuda_info = CUDAInfo()


class postfork:  # pylint: disable=invalid-name,too-few-public-methods
    def __init__(self, f: Callable[..., None]):
        if callable(f):
            self.wid = 0
            self.f = f
        else:
            self.f = None
            self.wid = f
        self.is_stub = True
        postfork_chain.append(self)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self.f:
            return self.f()
        self.f = args[0]
        return None


def find_next_device(key: str) -> int | None:
    for i in range(cuda_info.cuda_device_count):
        fetched_item = cache.get(f'{key}.{i}')
        current_app.logger.debug('looking up cache')
        current_app.logger.debug(f'{key}.{i}')
        current_app.logger.debug(fetched_item)
        if fetched_item is False or not fetched_item:
            cache.set(f'{key}.{i}', True)
            return i
    return None


def clear_device(engines_key: str, index: int):
    current_app.logger.debug('calling clear device')
    current_app.logger.debug(f'{engines_key}.{index}')
    cache.set(f'{engines_key}.{index}', False)


def call_next_available_pipeline(engines_key: str, *args: Any,
                                 **kwargs: Any) -> Any:
    global ENGINES  # pylint: disable=invalid-name,global-variable-not-assigned
    if (device_id := find_next_device(engines_key)) is None:
        return flask.Response(json.dumps({'error': 'Busy.'}), 503)
    try:
        ENGINES[engines_key][device_id]
    except IndexError:
        return flask.Response(
            json.dumps({
                'error':
                f'Invalid device ID. len(engines[{engines_key}]) = '
                f'{len(ENGINES[engines_key])}, device_id = {device_id}'
            }), 500)
    if not callable(ENGINES[engines_key][device_id]):  # Should not happen
        return flask.Response(
            json.dumps({
                'error':
                f'engines[{engines_key}][{device_id}] is not callable'
            }), 500)
    try:
        return ENGINES[engines_key][device_id](*args, **kwargs)
    except Exception as e:
        return flask.Response(json.dumps({'error': str(e)}), 500)
    finally:
        current_app.logger.debug('im actually clearing the engine')
        clear_device(engines_key, device_id)
