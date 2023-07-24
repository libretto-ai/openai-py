# Utilities for dealing with langchain

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Union, cast

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
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)


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
    messages: List[Union[BaseMessagePromptTemplate, BaseChatPromptTemplate, BaseMessage, List[Any]]]
) -> List[Dict]:
    """Format a chat template into something that Imaginary Programming can deal with"""
    lists = [_convert_message_to_dicts(message) for message in messages]
    # Flatten the list of lists
    flattened = [item for sublist in lists for item in sublist]
    return flattened


def _convert_message_to_dicts(
    message: Union[BaseMessagePromptTemplate, BaseChatPromptTemplate, BaseMessage, List[Any]]
) -> List[dict]:
    if isinstance(message, ChatMessage):
        return [{"role": message.role, "content": message.content}]
    elif isinstance(message, HumanMessage):
        return [{"role": "user", "content": message.content}]
    elif isinstance(message, AIMessage):
        formatted_messages = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            formatted_messages["function_call"] = message.additional_kwargs["function_call"]
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


def make_stub_inputs(inputs: Dict[str, Any], raw_lists=False) -> Dict[str, Any]:
    return make_stub_inputs_raw(inputs, "", raw_lists)  # type: ignore


def make_stub_inputs_raw(inputs: Any, prefix: str, raw_lists: bool):
    if isinstance(inputs, dict):
        dict_prefix = f"{prefix}." if prefix else ""
        return {
            k: make_stub_inputs_raw(v, f"{dict_prefix}{k}", raw_lists) for k, v in inputs.items()
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
        # return MessagesPlaceholder(
        #     variable_name=prefix
        # )
        if raw_lists:
            return [
                make_stub_inputs_raw(v, f"{prefix}[{i}]", raw_lists) for i, v in enumerate(inputs)
            ]
        return [ChatMessage(content=f"{{{prefix}}}", role="chat_history")]
    if isinstance(inputs, tuple):
        return tuple(
            make_stub_inputs_raw(v, f"{prefix}[{i}]", raw_lists) for i, v in enumerate(inputs)
        )
    resolved = make_stub_inputs_raw(format_langchain_value(inputs), prefix, raw_lists)
    return resolved


def format_completion_template_with_inputs(template: BasePromptTemplate, inputs: Dict[str, Any]):
    stub_inputs = make_stub_inputs(inputs, raw_lists=True)
    filtered_stub_inputs = {k: v for k, v in stub_inputs.items() if k in template.input_variables}
    if isinstance(template, BaseChatPromptTemplate):
        # We can't go through format_prompt because it doesn't like formatting the wrong types
        template_text = template.format(**filtered_stub_inputs)
    elif isinstance(template, StringPromptTemplate):
        template_text = template.format_prompt(**filtered_stub_inputs).to_string()
    else:
        raise ValueError(f"Unknown template type {type(template)}")
    return template_text


def format_chat_template_with_inputs(template, inputs):
    stub_inputs = make_stub_inputs(inputs, raw_lists=True)
    # Some of the templat formatters get upset if you pass in extra keys
    filtered_stub_inputs = {k: v for k, v in stub_inputs.items() if k in template.input_variables}
    if isinstance(template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        # We can't go through format_prompt because it doesn't like formatting the wrong types
        template_chat = template.format_messages(**filtered_stub_inputs)
    elif isinstance(template, StringPromptTemplate):
        template_chat = template.format_prompt(**filtered_stub_inputs).to_messages()
    else:
        raise ValueError(f"Unknown template type {type(template)}")
    return template_chat
