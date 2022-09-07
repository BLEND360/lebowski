# pylint: disable=import-outside-toplevel,global-statement
__all__ = ('app',)

from typing import Any
import json

from flask import Flask, request
from flask_caching import Cache
from uwsgidecorators import postfork
import flask

from .app_typing import EndpointRequestJSON

MODEL_ARGS = {
    'sum1': (('summarization',), {
        'model': 'google/pegasus-cnn_dailymail'
    }),
    'sum2': (('summarization',), {
        'model': 'tuner007/pegasus_paraphrase'
    }),
    'zsc': (('zero-shot-classification',), {
        'model': 'facebook/bart-large-mnli'
    })
}

app = Flask(__name__)
cache = Cache(app,
              config={
                  'CACHE_DIR': '/theapp/cache',
                  'CACHE_TYPE': 'FileSystemCache'
              })
cuda_device_count = 0  # pylint: disable=invalid-name
ENGINES: dict[str, list[Any]] = dict((key_, []) for key_ in MODEL_ARGS)


@postfork
def preload_engines():
    global ENGINES, cuda_device_count  # pylint: disable=invalid-name,global-variable-not-assigned
    # Setting the start method must be done before importing transformers.pipeines
    import torch
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    cuda_device_count = torch.cuda.device_count()
    from transformers.pipelines import pipeline
    for key, (args, kwargs) in MODEL_ARGS.items():
        for device_index in range(cuda_device_count):
            ENGINES[key].append(pipeline(*args, **kwargs, device=device_index))
            print('setting cache during init')
            print(f'{key}.{device_index}')
            cache.set(f'{key}.{device_index}', False)
    print('Done preloading engines')


def find_next_device(key: str) -> int | None:
    for i in range(cuda_device_count):
        fetched_item = cache.get(f'{key}.{i}')
        print('looking up cache')
        print(f'{key}.{i}')
        print(fetched_item)
        if fetched_item is False or not fetched_item:
            cache.set(f'{key}.{i}', True)
            return i
    return None


def clear_device(engines_key: str, index: int):
    print('calling clear device')
    print(f'{engines_key}.{index}')
    cache.set(f'{engines_key}.{index}', False)


def call_next_available_pipeline(engines_key: str, *args: Any,
                                 **kwargs: Any) -> Any:
    global ENGINES, last_device_index  # pylint: disable=invalid-name,global-variable-not-assigned
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
        print('im actually clearing the engine')
        clear_device(engines_key, device_id)


@app.route('/pegasus-summary', methods=['POST'])
def endpoint() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum1', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@app.route('/pegasus-paraphrase', methods=['POST'])
def paraphrase() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum2', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@app.route('/zero-shot-classification', methods=['POST'])
def zero_shot_classification() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        model_args = content['model_args'] if 'model_args' in content else {}
        return call_next_available_pipeline(
            'zsc', *(content['input'],), **{
                **model_args,
                **{
                    'multi_label': True
                }
            })
    return {'error': 'Invalid input'}


@app.route('/reset-cache')
def reset_cache() -> dict[str, bool]:
    values = {}
    for key, (_, __) in MODEL_ARGS.items():
        for i in range(cuda_device_count):
            cache_key = f'{key}.{i}'
            cache.set(f'{key}.{i}', False)
            values[cache_key] = False
    return values


@app.route('/reset-cache', methods=['POST'])
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


@app.route('/cache-status')
def cache_status() -> dict[str, bool]:
    values = {}
    for key, (_, __) in MODEL_ARGS.items():
        for i in range(cuda_device_count):
            cache_key = f'{key}.{i}'
            values[cache_key] = cache.get(f'{key}.{i}')
    return values
