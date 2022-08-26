from typing import TypedDict


class ModelArgs(TypedDict):
    max_length: int
    min_length: int


class EndpointRequestJSON(TypedDict):
    input: str | list[str]
    model_args: ModelArgs
