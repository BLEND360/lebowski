__all__ = ('app',)

from typing import Any

from flask import Flask, request
from transformers.pipelines import pipeline

from .app_typing import EndpointRequestJSON

app = Flask(__name__)


@app.route('/', methods=['POST'])
def endpoint() -> Any:
    if request.json:
        engine = pipeline('summarization',
                          model='google/pegasus-cnn_dailymail',
                          device=0)
        content: EndpointRequestJSON = request.json
        return engine(content['input'], content['model_args'])
    return None
