from typing import Any

from flask_caching import Cache

from .constants import MODEL_ARGS

__all__ = ('ENGINES', 'cache')

ENGINES: dict[str, list[Any]] = dict((key_, []) for key_ in MODEL_ARGS)
cache = Cache()
