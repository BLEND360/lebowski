# pylint: disable=wrong-import-position
__version__ = '0.1.0'
__all__ = ('create_app',)

# https://stackoverflow.com/questions/34145861/valueerror-failed-to-parse-cpython-sys-version-after-using-conda-command
import sys

sys.version = '3.10.1 (main, Aug 13 2022, 12:04:39) [GCC 11.3.0]'

from .main import create_app
