# Utilities for dealing with langchain

import dataclasses
import logging
import re
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Union, cast

from langchain.callbacks.manager import tracing_v2_callback_var
from langchain.prompts import (
    AIMessagePromptTemplate,
    BaseChatPromptTemplate,
    BasePromptTemplate,
    ChatMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.chat import (
    BaseChatPromptTemplate,
    BaseMessagePromptTemplate,
    BaseStringMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.agent import AgentAction

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
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if value is None:
        return None

    logger.warning("Cannot format value of type %s: %s", type(value), value)
    return value


def format_chat_template(
    messages: List[Union[BaseMessagePromptTemplate, BaseChatPromptTemplate, BaseMessage, List[Any]]]
) -> List[Dict[str, Any]]:
    """Format a chat template into something that Imaginary Programming can deal with"""
    lists = [_convert_message_to_dicts(message) for message in messages]
    # Flatten the list of lists
    return [item for sublist in lists for item in sublist]


def replace_array_variables_with_placeholders(raw_messages: List[dict], inputs: Dict[str, Any]):
    result = []
    next_index = 0
    for index, msg in enumerate(raw_messages):
        if next_index > index:
            continue

        if m := re.match(r"\{([a-zA-Z0-9_]+)\[0\].role\}", msg["role"]):
            var_name = m.group(1)
            input_list = inputs.get(var_name)
            if not isinstance(input_list, list):
                raise ValueError(f"Variable {var_name} is not a list")

            input_message_count = len(input_list)
            candidate_messages = raw_messages[index : index + input_message_count]

            # make sure we have enough messages in the output
            have_enough_messages = len(candidate_messages) == input_message_count
            all_messages_match_template = all(
                is_templated_chat_message(msg, var_name, candidate_msg_index)
                for candidate_msg_index, msg in enumerate(candidate_messages)
            )
            if have_enough_messages and all_messages_match_template:
                result.append({"role": "chat_history", "content": f"{{{var_name}}}"})
                next_index = index + input_message_count
                continue

        result.append(msg)
    return result


def is_templated_chat_message(msg: Dict[str, str], var_name: str, index: int) -> bool:
    return (
        msg["role"] == f"{{{var_name}[{index}].role}}"
        and msg["content"] == f"{{{var_name}[{index}].content}}"
    )


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
    elif isinstance(message, FakeInternalMessage):
        return [
            {
                "role": message.role,
                "content": message.content,
            }
        ]
    elif isinstance(message, dict):
        return [message]
    else:
        raise ValueError(f"Got unknown type {type(message)}: {message}")


class FakeInternalMessage(BaseMessage):
    role: str

    @property
    def type(self) -> str:
        return "foo"


def make_stub_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return make_stub_inputs_raw(inputs, "")  # type: ignore


def make_stub_inputs_raw(inputs: Dict[str, Any] | List[Any], prefix: str):
    if inputs is None:
        return f"{{{prefix}}}"
    if isinstance(inputs, dict):
        dict_prefix = f"{prefix}." if prefix else ""
        return {
            k: make_stub_inputs_raw(make_special_langchain_value(v, k), f"{dict_prefix}{k}")
            for k, v in inputs.items()
        }
    if isinstance(inputs, (str, int, float, bool)) or inputs is None:
        return f"{{{prefix}}}"
    if isinstance(inputs, list):
        return [make_stub_inputs_raw(v, f"{prefix}[{i}]") for i, v in enumerate(inputs)]
    if isinstance(inputs, tuple):
        return tuple(make_stub_inputs_raw(v, f"{prefix}[{i}]") for i, v in enumerate(inputs))
    if isinstance(inputs, ChatMessage):
        return {
            "role": make_stub_inputs_raw(inputs.role, f"{{{prefix}.role}}"),
            "content": make_stub_inputs_raw(inputs.content, f"{{{prefix}.content}}"),
        }
    if isinstance(inputs, BaseMessage):
        return FakeInternalMessage(
            role=f"{{{prefix}.role}}",
            content=f"{{{prefix}.content}}",
        )
    if isinstance(inputs, AgentAction):
        return AgentAction(
            tool=make_stub_inputs_raw(inputs.tool, f"{{{prefix}.tool}}"),
            tool_input=make_stub_inputs_raw(inputs.tool_input, f"{{{prefix}.tool_input}}"),
            log=make_stub_inputs_raw(inputs.log, f"{{{prefix}.log}}"),
        )
    return inputs


def make_special_langchain_value(v: Any, k: str):
    if k == "intermediate_steps" and not v:
        return [
            [
                AgentAction(
                    tool="{intermediate_steps}.tool",
                    tool_input="{intermediate_steps}.tool_input",
                    log="{intermediate_steps}.log",
                ),
                "{intermediate_steps[0][1]}",
            ]
        ]
    if k == "agent_history" and not v:
        # TODO: expand with placeholder?
        return [MessagesPlaceholder(variable_name="agent_history")]
    return v


def format_completion_template_with_inputs(template: BasePromptTemplate, inputs: Dict[str, Any]):
    stub_inputs = make_stub_inputs(inputs)
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
    stub_inputs = make_stub_inputs(inputs)
    return format_chat_template_with_stub_inputs(template, stub_inputs, "")


type2role = {"ai": "assistant", "human": "user", "system": "system"}


def format_chat_template_with_stub_inputs(
    template, stub_inputs, prefix: str
) -> List[Dict[str, Any]]:
    # Some of the templat formatters get upset if you pass in extra keys

    if isinstance(template, ChatPromptTemplate):
        # Container for a list of messages.
        # Introduces the `messages` property
        filtered_stub_inputs = {
            k: v for k, v in stub_inputs.items() if k in template.input_variables
        }
        sub_inputs = {**template.partial_variables, **stub_inputs}
        sub_messages = [
            format_chat_template_with_stub_inputs(m, sub_inputs, f"  {prefix}")
            for m in template.messages
        ]
        # Flatten sub_messages
        return [item for sublist in sub_messages for item in sublist]
    elif isinstance(template, PromptTemplate):
        filtered_stub_inputs = {
            k: v for k, v in stub_inputs.items() if k in template.input_variables
        }
        base_messages = template.format_prompt(**filtered_stub_inputs).to_messages()
        messages = [
            format_chat_template_with_stub_inputs(m, filtered_stub_inputs, f"  {prefix}")
            for m in base_messages
        ]
        # flatten
        return [item for sublist in messages for item in sublist]
    elif isinstance(template, MessagesPlaceholder):
        # declared inline with the special "chat_history" role
        return [{"role": "chat_history", "content": f"{{{template.variable_name}}}"}]
    elif isinstance(template, ChatMessagePromptTemplate):
        # Introduces the `role` property
        filtered_stub_inputs = {
            k: v for k, v in stub_inputs.items() if k in template.input_variables
        }
        return [
            {
                "role": template.role,
                "content": format_string_prompt_template(template.prompt, filtered_stub_inputs),
            }
        ]
    elif isinstance(template, BaseMessage):
        # Raw, untemplated message (AIMessage, HumanMessage, etc)
        return [
            {
                "role": getattr(template, "role", None) or type2role[template.type],
                "content": template.content,
            }
        ]
    elif isinstance(template, BaseStringMessagePromptTemplate):
        # introduces the `prompt`, but we just use format to make a string
        filtered_stub_inputs = {
            k: v for k, v in stub_inputs.items() if k in template.input_variables
        }
        messages = template.format_messages(**filtered_stub_inputs)
        sub_messages = [
            format_chat_template_with_stub_inputs(m, stub_inputs, f"  {prefix}") for m in messages
        ]
        # flatten
        return [item for sublist in sub_messages for item in sublist]
    elif isinstance(template, BaseMessagePromptTemplate):
        # A catch-all - all of the above should have caught this
        filtered_stub_inputs = {
            k: v for k, v in stub_inputs.items() if k in template.input_variables
        }
        messages = template.format_messages(**filtered_stub_inputs)
        sub_messages = [
            format_chat_template_with_stub_inputs(m, stub_inputs, f"  {prefix}") for m in messages
        ]
        return [item for sublist in sub_messages for item in sublist]
    else:  #  isinstance(template, BasePromptTemplate):
        raise ValueError(f"Unknown template type {type(template)}")


def format_string_prompt_template(v: StringPromptTemplate, stub_inputs: Dict[str, Any]) -> str:
    if isinstance(v, PromptTemplate):
        return v.template
    return v.format(**stub_inputs)
