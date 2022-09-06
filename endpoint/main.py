# pylint: disable=import-outside-toplevel,global-statement
__all__ = ('app',)

from typing import Any
import json

from flask import Flask, request
from flask_caching import Cache
from uwsgidecorators import postfork

import flask
try:
    import uwsgi  # pylint: disable=unused-import
except ImportError as e_:
    raise ImportError('Running outside of uWSGI is not supported') from e_

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
                  'CACHE_TYPE': 'uwsgi',
                  'CACHE_UWSGI_NAME': 'mycache@localhost'
              })
cuda_device_count = 0  # pylint: disable=invalid-name
ENGINES: dict[str, list[Any]] = dict((key_, []) for key_ in MODEL_ARGS)


@postfork
def preload_engines():
    global ENGINES, cuda_device_count  # pylint: disable=invalid-name,global-variable-not-assigned
    if len(ENGINES) == 0:
        # https://stackoverflow.com/questions/34145861/valueerror-failed-to-parse-cpython-sys-version-after-using-conda-command
        import sys
        sys.version = '3.10.1 (main, Aug 13 2022, 12:04:39) [GCC 11.3.0]'
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
                ENGINES[key].append(
                    pipeline(*args, **kwargs, device=device_index))
                cache.set(f'{key}@{device_index}', False)


def find_next_device(key: str) -> int | None:
    for i in range(cuda_device_count):
        if not cache.get(f'{key}@{i}'):
            cache.set(f'{key}@{i}', True)
            return i
    return None


def clear_device(engines_key: str, index: int):
    cache.set(f'{engines_key}@{index}', False)


def call_next_available_pipeline(engines_key: str, *args: Any,
                                 **kwargs: Any) -> Any:
    global ENGINES, last_device_index  # pylint: disable=invalid-name,global-variable-not-assigned
    if (device_id := find_next_device(engines_key)) is None:
        return flask.Response({'error': 'Busy.'}, 503)
    try:
        ENGINES[engines_key][device_id]
    except IndexError:
        return flask.Response(
            json.dumps({
                'error':
                f'Invalid device ID. len(engines[{engines_key}]) = '
                f'{len(ENGINES["sum1"])}, device_id = {device_id}'
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
        clear_device(engines_key, device_id)


@app.route('/', methods=['POST'])
def endpoint() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum1', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@app.route('/pegasus-paraphrase')
def paraphrase() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum2', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@app.route('/zero-shot-classification')
def zero_shot_classification():
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
