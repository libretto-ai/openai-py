# Utilities for dealing with langchain

import asyncio
import logging
import os
import time
import uuid
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

import aiohttp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import BaseMessagePromptTemplate, MessagesPlaceholder
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


def avoid_duplicate_call(f):
    """Simple decorator that prevents a function from being called with the same
    arguments twice in a row

    temporary until https://github.com/hwchase17/langchain/pull/7504 is fixed.
    """
    cache_key = None

    @wraps(f)
    def wrapper(*args, **kwargs):
        key = tuple(args) + tuple(kwargs.items())
        nonlocal cache_key
        if key != cache_key:
            cache_key = key
            return f(*args, **kwargs)
        return None

    return wrapper


def format_langchain_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [format_langchain_value(v) for v in value]
    return format_chat_template(value)


def format_chat_template(
    messages: List[Union[BaseMessagePromptTemplate, BaseMessage, List[Any]]]
) -> List[Dict]:
    """Format a chat template into something that Imaginary Programming can deal with"""
    lists = [_convert_message_to_dicts(message) for message in messages]
    # Flatten the list of lists
    return [item for sublist in lists for item in sublist]


def _convert_message_to_dicts(
    message: Union[BaseMessagePromptTemplate, BaseMessage, List[Any]]
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

    def _get_run(self, run_id):
        if run_id in self.runs:
            return self.runs[run_id]
        if run_id in self.parent_run_ids:
            return self._get_run(self.parent_run_ids[run_id])
        return None

    @avoid_duplicate_call
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

    @avoid_duplicate_call
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

    @avoid_duplicate_call
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

    @avoid_duplicate_call
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
        if self.template_text is None:
            return
        # TODO: need to generate a new event id for each prompt
        asyncio.run(
            self._async_send_completion(
                run_id, self.template_text, run["inputs"], prompts
            )
        )

    @avoid_duplicate_call
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
        asyncio.run(
            self._async_send_chat(run_id, self.template_chat, run["inputs"], messages)
        )

    @avoid_duplicate_call
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
        run = self.runs.get(run_id) or (parent_run_id and self.runs.get(parent_run_id))
        if not run:
            return
        now = time.time()
        response_time = (now - run["now"]) * 1000

        if "messages" in run:
            asyncio.run(
                self._async_send_chat(
                    run["prompt_event_id"],
                    self.template_chat,
                    run["inputs"],
                    run["messages"],
                    response,
                    response_time,
                )
            )
        elif "prompts" in run:
            asyncio.run(
                self._async_send_completion(
                    run["prompt_event_id"],
                    self.template_text,
                    run["inputs"],
                    run["prompts"],
                    response,
                    response_time,
                )
            )
        else:
            logger.warning("Missing prompts or messages in run %s %s", run_id, run)

    @avoid_duplicate_call
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""

    @avoid_duplicate_call
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
