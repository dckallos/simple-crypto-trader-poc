# test_no_validation.py

import pytest
from typed_defs import ChatCompletionMessageToolCallParam

def test_typedict_internals():
    """Demonstrates creating a ChatCompletionMessageToolCallParam
    object with no Pydantic validation. We simply assert fields or print them."""

    # Create a typed dict instance by raw assignment
    call: ChatCompletionMessageToolCallParam = {
        "id": "abc123",
        "type": "function",
        "function": {
            "name": "my_func",
            "arguments": '{"foo":"bar"}'
        }
    }

    # You can print or log the entire dictionary (object 'internals'):
    print("Raw typed dict =>", call)
    print("Function name =>", call["function"]["name"])
    print("Function args =>", call["function"]["arguments"])

    # Basic test: ensure the dictionary keys are present
    assert call["id"] == "abc123"
    assert call["type"] == "function"
    assert call["function"]["name"] == "my_func"
    assert call["function"]["arguments"] == '{"foo":"bar"}'
