from typing import Any
import json

from flask import Blueprint, request
import flask

from ..constants import MODEL_ARGS
from ..shared import cache
from ..utils import cuda_info

__all__ = ('cache_route_blueprint',)

cache_route_blueprint = Blueprint('cache_route_blueprint', __name__)


@cache_route_blueprint.route('/reset-cache')
def reset_cache() -> dict[str, bool]:
    values = {}
    for key, (_, __) in MODEL_ARGS.items():
        for i in range(cuda_info.cuda_device_count):
            cache_key = f'{key}.{i}'
            cache.set(f'{key}.{i}', False)
            values[cache_key] = False
    return values


@cache_route_blueprint.route('/reset-cache', methods=['POST'])
def reset_cache_json() -> Any:
    if (content := request.json):
        for key, val in content.items():
            if not cache.has(key):
                return flask.Response(
                    json.dumps({'error': f'Invalid key: {key}'}), 400)
            if not isinstance(val, bool):
                return flask.Response(
                    json.dumps({'error': f'Value at {key} is not a boolean'}),
                    400)
            cache.set(key, val)
    return cache_status()


@cache_route_blueprint.route('/cache-status')
def cache_status() -> dict[str, bool]:
    values = {}
    for key, (_, __) in MODEL_ARGS.items():
        for i in range(cuda_info.cuda_device_count):
            cache_key = f'{key}.{i}'
            values[cache_key] = cache.get(f'{key}.{i}')
    return values
