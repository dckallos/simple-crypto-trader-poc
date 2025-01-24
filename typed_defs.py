# typed_defs.py

from typing_extensions import Literal, Required, TypedDict

class Function(TypedDict, total=False):
    name: Required[str]
    arguments: Required[str]

class ChatCompletionMessageToolCallParam(TypedDict, total=False):
    id: Required[str]
    function: Required[Function]
    type: Required[Literal["function"]]
