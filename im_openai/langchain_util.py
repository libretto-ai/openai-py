# Utilities for dealing with langchain

import asyncio
import json
import logging
import os
import time
import uuid
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import aiohttp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.load.load import loads
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

from .client import send_event

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


class PromptWatchCallbacks(BaseCallbackHandler):
    runs: Dict[UUID, Dict[str, Any]] = {}
    project_key: str
    api_name: str
    template_chat: List[Dict] | None
    parent_run_ids: Dict[UUID, UUID] = {}

    def __init__(
        self,
        project_key: str,
        api_name: str,
        *,
        template_text: str | None = None,
        template_chat: List[Union[BaseMessagePromptTemplate, BaseMessage]]
        | None = None,
    ):
        self.project_key = project_key or os.environ.get("PROMPT_PROJECT_KEY", "")
        if not self.project_key:
            raise ValueError("project_key must be provided")
        self.runs = {}
        self.api_name = api_name
        if template_text is not None:
            self.template_text = template_text
        elif template_chat is not None:
            self.template_chat = format_chat_template(template_chat)
        else:
            self.template_chat = None
        self.parent_run_ids = {}

    def _get_run(self, run_id: UUID):
        if run_id in self.runs:
            return self.runs[run_id]
        if run_id in self.parent_run_ids:
            return self._get_run(self.parent_run_ids[run_id])
        return None

    def _get_run_metadata(self, run_id: UUID, metadata_key: str):
        """Walk up the parent chain looking for the nearest run with the metadata_key defined"""
        if run_id in self.runs and metadata_key in self.runs[run_id]:
            return self.runs[run_id][metadata_key]
        if run_id in self.parent_run_ids:
            return self._get_run_metadata(self.parent_run_ids[run_id], metadata_key)
        return None

    def on_chain_start(
        self,
        serialized,
        inputs,
        *a,
        run_id,
        parent_run_id,
        tags,
        metadata=None,
        **kwargs,
    ):
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_chain_start      %s [%s]",
            self._format_run_id(run_id),
            ", ".join(inputs.keys()),
        )

        self.runs[run_id] = {
            "inputs": inputs,
            "parent_run_id": parent_run_id,
            "prompt_event_id": uuid.uuid4(),
        }

        # super hack to extract the prompt if it exists
        prompt_template_chat = None
        prompt_template_text = None
        prompt_obj = serialized.get("kwargs", {}).get("prompt")
        if prompt_obj:
            prompt_template = loads(json.dumps(prompt_obj))
            if isinstance(prompt_template, BasePromptTemplate):
                self.runs[run_id]["prompt_template"] = prompt_template
        #         variable_inputs = {k: f"{{{k}}}" for k in prompt_template.input_variables}
        #         prompt_template_value = prompt_template.format_prompt(**variable_inputs)
        #         prompt_template_text = prompt_template_value.to_string()
        #         prompt_template_chat = prompt_template_value.to_messages()
        # self.runs[run_id]["prompt_template_text"] = prompt_template_text
        # self.runs[run_id]["prompt_template_chat"] = prompt_template_chat

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_agent_action     %s %s",
            self._format_run_id(run_id),
            action.tool,
        )
        """Run on agent action."""

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_agent_finish    %s %s",
            self._format_run_id(run_id),
            finish.log,
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_llm_start %s (%s prompts)",
            self._format_run_id(run_id),
            len(prompts),
        )

        run = self._get_run(run_id)
        if not run:
            logger.warning("on_llm_start Missing run %s", run_id)
            return
        run["prompts"] = prompts
        run["now"] = time.time()
        template = self._get_run_metadata(run_id, "prompt_template")
        template_text = (
            self._get_run_metadata(run_id, "prompt_template_text") or self.template_text
        )
        # TODO: need to generate a new event id for each prompt
        asyncio.run(
            self._async_send_completion(
                run_id,
                template_text,
                run["inputs"],
                prompts,
            )
        )

    def on_chat_model_start(
        self,
        serialized,
        messages: List[List[BaseMessage]],
        run_id,
        parent_run_id,
        tags,
        metadata=None,
        **kwargs,
    ):
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_chat_model_start %s (%s prompts)",
            self._format_run_id(run_id),
            len(messages),
        )
        run = self._get_run(run_id)
        if not run:
            return
        run["messages"] = messages
        run["now"] = time.time()

        template = self._get_run_metadata(run_id, "prompt_template")
        inputs = self._get_run_metadata(run_id, "inputs")
        stub_inputs = make_stub_inputs(inputs)

        template_chat = format_chat_template(template.format_messages(**stub_inputs))
        asyncio.run(
            self._async_send_chat(
                run_id,
                template_chat,
                run["inputs"],
                messages,
            )
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ):
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_llm_end          %s (%s responses)",
            self._format_run_id(run_id),
            len(response.generations),
        )
        # breakpoint()
        run = self._get_run(run_id)
        if not run:
            logger.warning("on_llm_end Missing run %s", run_id)
            return
        now = time.time()
        response_time = (now - run["now"]) * 1000

        if "messages" in run:
            template = self._get_run_metadata(run_id, "prompt_template")
            inputs = self._get_run_metadata(run_id, "inputs")
            stub_inputs = make_stub_inputs(inputs)

            template_chat = format_chat_template(
                template.format_messages(**stub_inputs)
            )
            asyncio.run(
                self._async_send_chat(
                    run["prompt_event_id"],
                    template_chat,
                    run["inputs"],
                    run["messages"],
                    response,
                    response_time,
                )
            )
        elif "prompts" in run:
            template_text = (
                self._get_run_metadata(run_id, "prompt_template_text")
                or self.template_text
            )
            asyncio.run(
                self._async_send_completion(
                    run["prompt_event_id"],
                    template_text,
                    run["inputs"],
                    run["prompts"],
                    response,
                    response_time,
                )
            )
        else:
            logger.warning("Missing prompts or messages in run %s %s", run_id, run)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if parent_run_id:
            self.parent_run_ids[run_id] = parent_run_id
        logger.info(
            "on_chain_end        %s [%s]",
            self._format_run_id(run_id),
            ", ".join(outputs.keys()),
        )
        if run_id in self.runs:
            del self.runs[run_id]

    async def _async_send_completion(
        self,
        prompt_event_id: UUID,
        template_text: str,
        inputs: Dict,
        prompts: List[str],
        result: LLMResult | None = None,
        response_time: float | None = None,
    ):
        async with aiohttp.ClientSession() as session:
            generations_lists = result.generations if result else []
            for prompt, generations in zip_longest(prompts, generations_lists):
                response_text = (
                    "".join(g.text for g in generations) if generations else None
                )
                json_inputs = {
                    key: format_langchain_value(value) for key, value in inputs.items()
                }
                # If no template is passed in, pass in the completion prompt instead
                prompt = {"completion": prompt} if not template_text else None
                # TODO: gather these up and send them all at once
                await send_event(
                    session,
                    project_key=self.project_key,
                    api_name=self.api_name,
                    prompt_template_text=template_text,
                    prompt_template_chat=None,
                    prompt_params=json_inputs,
                    prompt_event_id=str(prompt_event_id),
                    chat_id=None,
                    response=response_text,
                    response_time=response_time,
                )

    async def _async_send_chat(
        self,
        prompt_event_id: UUID,
        template_chats: List[Dict] | None,
        inputs: Dict,
        messages_list: List[List[BaseMessage]],
        result: LLMResult | None = None,
        response_time: float | None = None,
    ):
        async with aiohttp.ClientSession() as session:
            generations_lists = result.generations if result else []
            for messages, generations in zip_longest(messages_list, generations_lists):
                response_text = (
                    "".join(g.text for g in generations) if generations else None
                )
                json_inputs = {
                    key: format_langchain_value(value) for key, value in inputs.items()
                }
                # If no template is passed in, pass in the prompt instead
                prompt = (
                    {"chat": format_chat_template(messages)}  # type: ignore
                    if not template_chats
                    else None
                )
                # TODO: gather these up and send them all at once
                await send_event(
                    session,
                    project_key=self.project_key,
                    api_name=self.api_name,
                    prompt_template_text=None,
                    prompt_template_chat=template_chats,
                    prompt_params=json_inputs,
                    prompt_event_id=str(prompt_event_id),
                    chat_id=None,
                    response=response_text,
                    response_time=response_time,
                    prompt=prompt,
                )

    def _format_run_id(self, run_id: UUID) -> str:
        """Generates a hierarchy of run_ids by recursively walking self.parent_ids"""
        if run_id in self.parent_run_ids:
            return f"{self._format_run_id(self.parent_run_ids[run_id])} -> {run_id}"
        return str(run_id)


def make_stub_inputs(inputs: Any, prefix=""):
    if isinstance(inputs, dict):
        dict_prefix = f"{prefix}." if prefix else ""
        return {
            k: make_stub_inputs(v, prefix=f"{dict_prefix}{k}")
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
            make_stub_inputs(v, prefix=f"{prefix}[{i}]") for i, v in enumerate(inputs)
        ]
    if isinstance(inputs, tuple):
        return tuple(
            make_stub_inputs(v, prefix=f"{prefix}[{i}]") for i, v in enumerate(inputs)
        )
    resolved = make_stub_inputs(format_langchain_value(inputs), prefix=prefix)
    return resolved
