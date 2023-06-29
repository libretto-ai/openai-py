# Utilities for dealing with langchain

from typing import Any, List, Union

from langchain.prompts.chat import BaseMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)


def format_chat_template(
    messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]
) -> List[Any]:
    """Format a chat template into something that Imaginary Programming can deal with"""
    lists = [_convert_message_to_dicts(message) for message in messages]
    # Flatten the list of lists
    return [item for sublist in lists for item in sublist]


def _convert_message_to_dicts(
    message: Union[BaseMessagePromptTemplate, BaseMessage]
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
    elif isinstance(message, BaseMessagePromptTemplate):
        vars = message.input_variables
        # create a fake dictionary mapping name -> '{name}'
        vars_as_templates = {v: f"{{{v}}}" for v in vars}
        formatted_messages = message.format_messages(**vars_as_templates)
        return format_chat_template(formatted_messages)
    else:
        raise ValueError(f"Got unknown type {message}")
