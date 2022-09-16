from typing import Any

from flask import Blueprint, request

from ..app_typing import EndpointRequestJSON
from ..utils import call_next_available_pipeline

__all__ = ('model_route_blueprint',)

model_route_blueprint = Blueprint('model_route_blueprint', __name__)


@model_route_blueprint.route('/pegasus-summary', methods=['POST'])
def endpoint() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum1', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@model_route_blueprint.route('/pegasus-paraphrase', methods=['POST'])
def paraphrase() -> Any:
    if request.json:
        content: EndpointRequestJSON = request.json
        return call_next_available_pipeline(
            'sum2', *(content['input'],),
            **(content['model_args'] if 'model_args' in content else {}))
    return {'error': 'Invalid input'}


@model_route_blueprint.route('/zero-shot-classification', methods=['POST'])
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
