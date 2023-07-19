# Utilities for dealing with langchain

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from itertools import zip_longest
from typing import Any, Dict, List, Optional, TypeVar, Union, cast
from uuid import UUID

import aiohttp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import tracing_v2_callback_var
from langchain.prompts import (
    BaseChatPromptTemplate,
    BasePromptTemplate,
    StringPromptTemplate,
)
from langchain.prompts.chat import (
    BaseChatPromptTemplate,
    BaseMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    LLMResult,
    SystemMessage,
)

from im_openai import client

from .patch import loads

logger = logging.getLogger(__name__)

from functools import wraps


def format_langchain_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [format_langchain_value(v) for v in value]
    if isinstance(value, dict):
        return {k: format_langchain_value(v) for k, v in value.items()}
    if isinstance(value, StringPromptTemplate):
        inputs = {k: f"{{{k}}}" for k in value.input_variables}
        return value.format(**inputs)
    if isinstance(value, BaseMessage):
        return format_chat_template([value])[0]
    return format_chat_template(value)


def format_chat_template(
    messages: List[
        Union[BaseMessagePromptTemplate, BaseChatPromptTemplate, BaseMessage, List[Any]]
    ]
) -> List[Dict]:
    """Format a chat template into something that Imaginary Programming can deal with"""
    lists = [_convert_message_to_dicts(message) for message in messages]
    # Flatten the list of lists
    return [item for sublist in lists for item in sublist]


def _convert_message_to_dicts(
    message: Union[
        BaseMessagePromptTemplate, BaseChatPromptTemplate, BaseMessage, List[Any]
    ]
) -> List[dict]:
    if isinstance(message, ChatMessage):
        return [{"role": message.role, "content": message.content}]
    elif isinstance(message, HumanMessage):
        return [{"role": "user", "content": message.content}]
    elif isinstance(message, AIMessage):
        formatted_messages = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            formatted_messages["function_call"] = message.additional_kwargs[
                "function_call"
            ]
        return [formatted_messages]
    elif isinstance(message, SystemMessage):
        return [{"role": "system", "content": message.content}]
    elif isinstance(message, MessagesPlaceholder):
        formatted_messages = {
            "role": "chat_history",
            "content": f"{{{message.variable_name}}}",
        }
        return [formatted_messages]
    elif isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        vars = message.input_variables
        # create a fake dictionary mapping name -> '{name}'
        vars_as_templates = {v: f"{{{v}}}" for v in vars}
        formatted_messages = message.format_messages(**vars_as_templates)
        return format_chat_template(formatted_messages)  # type: ignore
    elif isinstance(message, dict):
        return [message]
    else:
        raise ValueError(f"Got unknown type {type(message)}: {message}")


def make_stub_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return make_stub_inputs_raw(inputs, "")  # type: ignore


def make_stub_inputs_raw(inputs: Any, prefix: str):
    if isinstance(inputs, dict):
        dict_prefix = f"{prefix}." if prefix else ""
        return {
            k: make_stub_inputs_raw(v, prefix=f"{dict_prefix}{k}")
            for k, v in inputs.items()
        }
    if isinstance(inputs, (str, int, float, bool)):
        return f"{{{prefix}}}"
    if isinstance(inputs, list):
        # TODO: figure out a way to collapse lists. Right now this will create stuff like:
        # [
        #     {
        #         "role": "{agent_history[0].role}",
        #         "content": "{agent_history[0].content}",
        #     },
        #     {
        #         "role": "{agent_history[1].role}",
        #         "content": "{agent_history[1].content}",
        #     },
        # ]
        # return [f"{{{prefix}}}"] * len(inputs)
        return [
            make_stub_inputs_raw(v, prefix=f"{prefix}[{i}]")
            for i, v in enumerate(inputs)
        ]
    if isinstance(inputs, tuple):
        return tuple(
            make_stub_inputs_raw(v, prefix=f"{prefix}[{i}]")
            for i, v in enumerate(inputs)
        )
    resolved = make_stub_inputs_raw(format_langchain_value(inputs), prefix=prefix)
    return resolved
