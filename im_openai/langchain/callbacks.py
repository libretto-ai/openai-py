import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from itertools import zip_longest
from typing import Any, Dict, List, Optional, TypedDict, Union, _SpecialForm, cast
from uuid import UUID

import aiohttp
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import tracing_v2_callback_var
from langchain.load.load import load
from langchain.prompts import BasePromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult

from im_openai import client

from . import util

logger = logging.getLogger(__name__)

from functools import wraps


class RunInfo(TypedDict, total=False):
    prompt_template_name: Optional[str]
    inputs: Dict[str, Any]
    parent_run_id: Optional[UUID]
    prompt_event_id: UUID
    prompt_template: Optional[Any]
    api_name: Optional[Optional[str]]
    now: Optional[float]
    prompts: Optional[List[str]]
    model_params: Dict[str, Any]
    messages: Optional[List[List[BaseMessage]]]


def record_run_info(run_type: str):
    """Decorator to record the run info of a function call"""

    def _record_decorator(func):
        @wraps(func)
        def record(self, *args, **kwargs):
            if "run_id" in kwargs:
                run_id = kwargs["run_id"]
                prev_type = self.run_types.get(run_id, None)
                if prev_type is not None and prev_type != run_type:
                    logger.warning(f"converting run {run_id} from {prev_type} to {run_type}")
                self.run_types[kwargs["run_id"]] = run_type
                if "parent_run_id" in kwargs:
                    parent_run_id = kwargs["parent_run_id"]
                    if parent_run_id:
                        self.parent_run_ids[run_id] = parent_run_id
                    elif run_type != "chain":
                        logger.warning(f"WARNING: parent_run_id is None for {run_type} {run_id}")

            return func(self, *args, **kwargs)

        return record

    return _record_decorator


class PromptWatchCallbacks(BaseCallbackHandler):
    api_key: str
    project_key: str | None
    prompt_template_name: str | None

    template_chat: List[Dict] | None
    template_text: str | None

    chat_id: str | None = None

    runs: Dict[UUID, RunInfo] = {}
    """A dictionary of information about a run, keyed by run_id"""
    parent_run_ids: Dict[UUID, UUID] = {}
    """In case self.runs is missing a run, we can walk up the parent chain to find it"""

    run_types: Dict[UUID, str] = {}

    server_event_ids: Dict[UUID, client.SendEventResponse | None] = {}
    """Mapping of run_id to server event id"""

    valid_namespaces: List[str] | None = None

    # uncomment to debug with pdb
    # raise_error = True

    def __init__(
        self,
        *,
        api_key: str,
        api_name: str | None = None,
        prompt_template_name: str | None = None,
        project_key: Optional[str] = None,
        template_text: Optional[str] = None,
        template_chat: Optional[List[Union[BaseMessagePromptTemplate, BaseMessage]]] = None,
        valid_namespaces: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize the callback handler

        Args:
            api_key (str): The Imaginary Programming API key
            project_key (str): The Imaginary Programming project key
            prompt_template_name (str): The Imaginary Programming API name
            template_text (Optional[str], optional): The template to use for completion events. Defaults to None.
            template_chat (Optional[List[Union[BaseMessagePromptTemplate, BaseMessage]]], optional): The template to use for chat events. Defaults to None.
        """
        self.project_key = project_key or os.environ.get("PROMPT_PROJECT_KEY", None)
        self.api_key = api_key or os.environ.get("PROMPT_API_KEY", "")
        if not self.api_key:
            raise ValueError("api_key must be provided")
        self.valid_namespaces = valid_namespaces
        self.runs = {}
        self.prompt_template_name = prompt_template_name or api_name
        self.chat_id = chat_id
        if template_text is not None:
            self.template_text = template_text
        elif template_chat is not None:
            self.template_chat = util.format_chat_template(cast(Any, template_chat))
        else:
            self.template_chat = None
        self.parent_run_ids = {}

    def _get_run(self, run_id: UUID):
        if run_id in self.runs:
            return self.runs[run_id]
        if run_id in self.parent_run_ids:
            return self._get_run(self.parent_run_ids[run_id])
        return None

    def _get_run_info(self, run_id: UUID, metadata_key: str):
        """Walk up the parent chain looking for the nearest run with the metadata_key defined"""
        if run_id in self.runs and metadata_key in self.runs[run_id]:
            return self.runs[run_id][metadata_key]
        if run_id in self.parent_run_ids:
            return self._get_run_info(self.parent_run_ids[run_id], metadata_key)
        return None

    def _get_server_event_id(self, run_id: UUID):
        """Lazily insert a server event id if it doesn't exist. This will get filled in later when the initial event is sent"""
        if run_id not in self.server_event_ids:
            self.server_event_ids[run_id] = None
        return self.server_event_ids[run_id]

    @record_run_info("chain")
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        logger.info(
            "on_chain_start      %s [%s] %s",
            self._format_run_id(run_id),
            ", ".join(inputs.keys()),
            ".".join(serialized["id"]),
        )

        run: RunInfo = {
            "inputs": inputs,
            "parent_run_id": parent_run_id,
            "prompt_event_id": uuid.uuid4(),
            "model_params": {},
            "prompt_template_name": None,
            "api_name": None,
            "messages": [],
            "now": None,
            "prompt_template": None,
            "prompts": [],
        }
        self.runs[run_id] = run

        # First try to load the chain
        prompt_template = self._extract_template_from_chain(serialized)

        if isinstance(prompt_template, BasePromptTemplate):
            self.runs[run_id]["prompt_template"] = prompt_template

            # User might have overridden the prompt_template_name
            prompt_template_name: str | None = prompt_template._lc_kwargs.get(
                "additional_kwargs", {}
            ).get("ip_prompt_template_name")
            self.runs[run_id]["api_name"] = prompt_template_name
        elif prompt_template is None:
            logger.warning("Missing prompt template for chain %s", ".".join(serialized["id"]))
        else:
            logger.warning("Unrecognized prompt template type: %s", type(prompt_template))

    def _extract_template_from_chain(self, serialized: Dict[str, Any]):
        """Attempt to extract the prompt template from a serialized
        representation of a chain. Tries first to use standard deserialization,
        then falls back to a hacky method of extracting the prompt from the
        kwargs"""
        prompt_template = None
        try:
            chain = load(serialized, valid_namespaces=self.valid_namespaces)
            prompt_template = chain.prompt
        except (NotImplementedError, AttributeError) as e:
            module_path = cast(list, serialized.get("id", []))
            logger.debug("Skipping chain derserialization %s: %s", ".".join(module_path), e)

            # super hack to extract the prompt if it exists
            prompt_obj = serialized.get("kwargs", {}).get("prompt")
            if prompt_obj:
                prompt_template = load(prompt_obj, valid_namespaces=self.valid_namespaces)

        return prompt_template

    @record_run_info("chain")
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        logger.info(
            "on_agent_action     %s %s",
            self._format_run_id(run_id),
            action.tool,
        )
        """Run on agent action."""

    @record_run_info("chain")
    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        logger.info(
            "on_agent_finish     %s %s",
            self._format_run_id(run_id),
            finish.log,
        )

    @record_run_info("tool")
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        logger.info(
            "on_tool_start       %s",
            self._format_run_id(run_id),
        )

    @record_run_info("retriever")
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        logger.info(
            "on_retriever_start    %s",
            self._format_run_id(run_id),
        )

    @record_run_info("llm")
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
        """Run when a text-based LLM runs"""
        logger.info(
            "on_llm_start %s (%s prompts)",
            self._format_run_id(run_id),
            len(prompts),
        )

        run = self._get_run(run_id)
        if not run:
            logger.warning("on_llm_start Missing run %s", run_id)
            return
        model_params = _extract_openai_params(serialized)
        run["prompts"] = prompts
        run["now"] = time.time()
        run["model_params"] = model_params

    @record_run_info("llm")
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
        """Runs when a chat-based LLM runs"""
        logger.info(
            "on_chat_model_start %s (%s prompts)",
            self._format_run_id(run_id),
            len(messages),
        )

        run = self._get_run(run_id)
        if not run:
            return
        model_params = _extract_openai_params(serialized)
        run["messages"] = messages
        run["now"] = time.time()
        run["model_params"] = model_params

    @record_run_info("llm")
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Runs when either a text-based or chat-based LLM ends"""
        logger.info(
            "on_llm_end          %s (%s responses)",
            self._format_run_id(run_id),
            len(response.generations),
        )

        run = self._get_run(run_id)
        if not run:
            logger.warning("on_llm_end Missing run %s", run_id)
            return
        now = time.time()
        model_params = run["model_params"]
        response_time = None
        if "now" in run:
            response_time = (now - cast(float, run["now"])) * 1000
        prompt_template_name = self._get_run_info(run_id, "api_name")

        if "messages" in run:
            template_chat = self._resolve_chat_template(run_id)
            self._send_chat(
                run_id,
                template_chat,
                run["inputs"],
                run["messages"] or [],
                model_params,
                response,
                response_time,
                parent_event_id=parent_run_id,
                prompt_template_name=prompt_template_name,
                chat_id=self.chat_id,
            )
        elif "prompts" in run:
            template_text = self._get_run_info(run_id, "prompt_template_text") or self.template_text
            prompt_template_name = self._get_run_info(run_id, "api_name")
            self._send_completion(
                run_id,
                template_text,
                run["inputs"],
                run["prompts"] or [],
                parent_event_id=parent_run_id,
                model_params=model_params,
                result=response,
                response_time=response_time,
                prompt_template_name=prompt_template_name,
                chat_id=self.chat_id,
            )
        else:
            logger.warning("Missing prompts or messages in run %s %s", run_id, run)

    @record_run_info("llm")
    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""

    @record_run_info("chain")
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        logger.info(
            "on_chain_end        %s [%s]",
            self._format_run_id(run_id),
            ", ".join(outputs.keys()),
        )
        if run_id in self.runs:
            del self.runs[run_id]

    def _send_event(
        self,
        run_id: UUID,
        template: List[Dict] | str | None,
        inputs: Dict,
        iterations: List[str] | List[List[BaseMessage]],
        *,
        result: LLMResult | None = None,
        response_time: float | None = None,
        parent_event_id: UUID | None = None,
        model_params: Dict | None = None,
        prompt_template_name: str | None = None,
        chat_id: str | None = None,
        is_chat: bool = False,
    ):
        # walk up the parent chain until we hit the root
        parent_event_id = run_id
        while parent_event_id in self.parent_run_ids:
            parent_event_id = self.parent_run_ids[parent_event_id]
        generations_lists = result.generations if result else []
        for iteration, generations in zip_longest(iterations, generations_lists):
            response_text = "".join(g.text for g in generations) if generations else None
            json_inputs = {key: util.format_langchain_value(value) for key, value in inputs.items()}
            model_params = model_params.copy() if model_params else {}
            if is_chat:
                model_params["modelType"] = "chat"
                prompt = (
                    {"chat": util.format_chat_template(iteration)}  # type: ignore
                    if not template
                    else None
                )
                prompt_template_text = None
                prompt_template_chat = template
            else:
                model_params["modelType"] = "completion"
                prompt = {"completion": iteration} if not template else None
                prompt_template_text = template
                prompt_template_chat = None

            # TODO: gather these up and send them all at once
            prompt_event_id = self._get_server_event_id(run_id)
            response = client.send_event_background(
                project_key=self.project_key,
                api_key=self.api_key,
                prompt_template_name=prompt_template_name or self.prompt_template_name,
                prompt_template_text=prompt_template_text,  # type: ignore
                prompt_template_chat=prompt_template_chat,  # type: ignore
                prompt_params=json_inputs,
                prompt_event_id=prompt_event_id["id"] if prompt_event_id else None,
                model_params=model_params,
                chat_id=chat_id,
                response=response_text,
                response_time=response_time,
                prompt=prompt,
                parent_event_id=str(parent_event_id) if parent_event_id else None,
            )
            if response:
                self.server_event_ids[run_id] = response

    def _send_chat(
        self,
        run_id: UUID,
        template_chats: List[Dict] | None,
        inputs: Dict,
        messages_list: List[List[BaseMessage]],
        model_params: Dict | None = None,
        result: LLMResult | None = None,
        response_time: float | None = None,
        parent_event_id: UUID | None = None,
        prompt_template_name: str | None = None,
        chat_id: str | None = None,
    ):
        self._send_event(
            run_id,
            template_chats,
            inputs,
            messages_list,
            result=result,
            response_time=response_time,
            parent_event_id=parent_event_id,
            model_params=model_params,
            prompt_template_name=prompt_template_name,
            chat_id=chat_id,
            is_chat=True,
        )

    def _send_completion(
        self,
        prompt_event_id: UUID,
        template_text: str | None,
        inputs: Dict,
        prompts: List[str],
        *,
        parent_event_id: UUID | None = None,
        model_params: Dict | None = None,
        result: LLMResult | None = None,
        response_time: float | None = None,
        prompt_template_name: str | None = None,
        chat_id: str | None = None,
    ):
        self._send_event(
            prompt_event_id,
            template_text,
            inputs,
            prompts,
            result=result,
            response_time=response_time,
            parent_event_id=parent_event_id,
            model_params=model_params,
            prompt_template_name=prompt_template_name,
            chat_id=chat_id,
            is_chat=False,
        )

    def _format_run_id(self, run_id: UUID) -> str:
        """Generates a hierarchy of run_ids by recursively walking self.parent_ids"""
        if run_id in self.parent_run_ids:
            return f"{self._format_run_id(self.parent_run_ids[run_id])} -> {run_id}"
        return str(run_id)

    def _resolve_chat_template(self, run_id: UUID):
        """Resolve the template_chat into a list of dicts"""
        template: Optional[BasePromptTemplate] = self._get_run_info(run_id, "prompt_template")
        inputs: Optional[Dict[str, Any]] = self._get_run_info(run_id, "inputs")
        template_chat = None
        if template and inputs:
            messages = util.format_chat_template_with_inputs(template, inputs)
            template_chat = util.replace_array_variables_with_placeholders(messages, inputs)
        return template_chat

    def _resolve_completion_template(self, run_id: UUID):
        """Resolve the template_chat into a list of dicts"""
        template = self._get_run_info(run_id, "prompt_template")
        inputs = self._get_run_info(run_id, "inputs")
        template_text: str | None = None
        if template and inputs:
            template_text = util.format_completion_template_with_inputs(template, inputs)
            # template_text = format_chat_template(template_text)  # type: ignore
        return template_text


def _extract_openai_params(serialized: Dict):
    return {k: v for k, v in serialized["kwargs"].items() if isinstance(v, (int, float, str, bool))}


def enable_prompt_watch_tracing(*args, **kwargs):
    """Manually enable prompt watch tracing, returns the previous callback handler to be used when disabling"""
    callbacks = PromptWatchCallbacks(*args, **kwargs)
    old_tracing_v2_callback = tracing_v2_callback_var.get()
    tracing_v2_callback_var.set(cast(Any, callbacks))
    return old_tracing_v2_callback


def disable_prompt_watch_tracing(old_callbacks: BaseCallbackHandler | None):
    """Manually disable prompt watch tracing, possibly restoring the previous callback handler"""
    tracing_v2_callback_var.set(cast(Any, old_callbacks))


@contextmanager
def prompt_watch_tracing(*args, **kwargs):
    """Inject a tracing callback into langchain to watch for prompts.

    Note that this hijacks the v2 tracing callback, so if you're using that for something else, this will break it.

    usage::

        with prompt_watch_tracing(api_key, api_name, template_text="Hello {name}"):
            chain = LLMChain(llm=...)
            chain.run("Hello world", inputs={"name": "world"})
    """
    old_tracing_v2_callback = enable_prompt_watch_tracing(*args, **kwargs)
    yield
    disable_prompt_watch_tracing(old_tracing_v2_callback)
