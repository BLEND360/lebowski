# pylint: disable=import-outside-toplevel,global-statement
__all__ = ('create_app',)

import logging

from botocore.exceptions import NoCredentialsError
from flask import Flask
import watchtower

from .constants import MODEL_ARGS
from .shared import ENGINES, cache
from .utils import cuda_info

try:
    from uwsgidecorators import postfork
    postfork_chain = []
    postfork_is_stub = False  # pylint: disable=invalid-name
except ImportError:
    postfork_is_stub = True  # pylint: disable=invalid-name
    from .utils import postfork, postfork_chain


def create_app() -> Flask:
    app = Flask(__name__)
    cache.init_app(app,
                   config={
                       'CACHE_DIR': '.cache',
                       'CACHE_TYPE': 'FileSystemCache',
                       'CACHE_DEFAULT_TIMEOUT': 0,
                       'CACHE_THRESHOLD': 0
                   })
    try:
        handler = watchtower.CloudWatchLogHandler(log_group_name=app.name)
        app.logger.addHandler(handler)  # pylint: disable=no-member
        logging.getLogger('werkzeug').addHandler(handler)
        logging.getLogger('endpoint.preload').addHandler(handler)
    except NoCredentialsError:
        pass
    from .routes import cache_route_blueprint, model_route_blueprint
    app.register_blueprint(cache_route_blueprint)
    app.register_blueprint(model_route_blueprint)
    if postfork_is_stub:
        for f in postfork_chain:
            f()
    return app


@postfork
def preload_engines():
    global ENGINES  # pylint: disable=invalid-name,global-variable-not-assigned
    # Setting the start method must be done before importing transformers.pipeines
    import torch
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    from transformers.pipelines import pipeline
    cuda_info.cuda_device_count = torch.cuda.device_count()
    logger = logging.getLogger('endpoint.preload')
    for key, (args, kwargs) in MODEL_ARGS.items():
        for device_index in range(cuda_info.cuda_device_count):
            ENGINES[key].append(pipeline(*args, **kwargs, device=device_index))
            logger.debug('setting cache during init')
            logger.debug('%s.%s', key, device_index)
            cache.set(f'{key}.{device_index}', False)
    logger.debug('Done preloading engines')
