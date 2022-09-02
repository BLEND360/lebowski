# pylint: disable=import-outside-toplevel,global-statement
__all__ = ('app',)

from os.path import isfile
from typing import Any
import json
import sqlite3
from uwsgidecorators import postfork

from flask import Flask, request

import flask

from .app_typing import EndpointRequestJSON

app = Flask(__name__)
engines: list[Any] = []
last_device_index = 0  # pylint: disable=invalid-name

DATABASE_FILE = './db.sqlite'

need_schema = not isfile(DATABASE_FILE)
if need_schema:
    db = sqlite3.connect(DATABASE_FILE).cursor().execute(
        '''CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              device_id INT NOT NULL CHECK(device_id >= 0),
                              completed INT NOT NULL CHECK(completed == 0 || completed == 1),
                              date_created TEXT NOT NULL)''')


@postfork
def preload_engine():
    global engines, last_device_index  # pylint: disable=invalid-name,global-variable-not-assigned
    if len(engines) == 0:
        # https://stackoverflow.com/questions/34145861/valueerror-failed-to-parse-cpython-sys-version-after-using-conda-command
        import sys
        sys.version = '3.10.1 (main, Aug 13 2022, 12:04:39) [GCC 11.3.0]'
        # Setting the start method must be done before importing transformers.pipeines
        import torch
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
        last_device_index = torch.cuda.device_count() - 1
        from transformers.pipelines import pipeline
        for i in range(torch.cuda.device_count()):
            engines.append(
                pipeline('summarization',
                         model='google/pegasus-cnn_dailymail',
                         device=i))


@app.route('/', methods=['POST'])
def endpoint() -> Any:
    global engine, last_device_index  # pylint: disable=invalid-name,global-variable-not-assigned
    if request.json:
        with sqlite3.connect(DATABASE_FILE) as conn:
            cur = conn.cursor()
            cur.execute('SELECT device_id FROM jobs ORDER BY id DESC LIMIT 1')
            device_id = (0 if not (result := cur.fetchone()) else
                         (0 if result[0] == last_device_index else result[0] +
                          1))
            cur.execute(
                '''INSERT INTO jobs(device_id, completed, date_created)
            VALUES (?, 0, datetime("now"))''', (device_id,))
            job_id = cur.execute('SELECT LAST_INSERT_ROWID()').fetchone()[0]
        content: EndpointRequestJSON = request.json
        try:
            engines[device_id]
        except IndexError:
            return flask.Response(
                json.dumps({
                    'error':
                    f'Invalid device ID. len(engines) = {len(engines)}, device_id = {device_id}'
                }), 500)
        if not callable(engines[device_id]):  # Should not happen
            return flask.Response(
                json.dumps({'error': f'engines[{device_id}] is not callable'}),
                500)
        try:
            return engines[device_id](
                content['input'],
                **(content['model_args'] if 'model_args' in content else {}))
        except Exception as e:
            return flask.Response(json.dumps({'error': str(e)}), 500)
        finally:
            with sqlite3.connect(DATABASE_FILE) as conn:
                cur = conn.cursor()
                cur.execute('UPDATE jobs SET completed = 1 WHERE id = ?',
                            (job_id,))
    return {'error': 'Invalid input'}
