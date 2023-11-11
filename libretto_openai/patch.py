import json
import logging
import os
import uuid
from contextlib import contextmanager
from itertools import tee
import types
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypedDict, cast

import openai
from openai.types.completion import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.resources import Completions
from openai.resources.chat import Completions as ChatCompletions

from .client import event_session
from .template import TemplateChat, TemplateString
from .pii import Redactor


logger = logging.getLogger(__name__)


class LibrettoCreateParamDict(TypedDict):
    project_key: str | None
    api_key: str | None
    prompt_template_name: str | None
    api_name: str | None
    event_id: str | None
    template_text: str | None
    template_chat: List | None
    template_params: Any | None
    chat_id: str | None
    parent_event_id: str | None
    feedback_key: str | None


# This is a helper function that allows for instantiating a LibrettoCreateParamDict
# without the need for specifying every key, which is required by TypedDict in
# Python < 3.11 (NotRequired was added to address this).
def LibrettoCreateParams(  # pylint: disable=invalid-name
    project_key: str | None = None,
    api_key: str | None = None,
    prompt_template_name: str | None = None,
    api_name: str | None = None,
    event_id: str | None = None,
    template_text: str | None = None,
    template_chat: List | None = None,
    template_params: Any | None = None,
    chat_id: str | None = None,
    parent_event_id: str | None = None,
    feedback_key: str | None = None,
):
    return LibrettoCreateParamDict(
        project_key=project_key,
        api_key=api_key,
        prompt_template_name=prompt_template_name,
        api_name=api_name,
        event_id=event_id,
        template_text=template_text,
        template_chat=template_chat,
        template_params=template_params,
        chat_id=chat_id,
        parent_event_id=parent_event_id,
        feedback_key=feedback_key,
    )


def _patch_openai_completions(
    cls: Completions | ChatCompletions,
    get_prompt_template: Callable,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
    api_key: Optional[str] = None,
    allow_unnamed_prompts: bool = False,
    redact_pii: bool = False,
):
    """Patch an openai class to add logging capabilities.

    Returns a function which may be called to "unpatch" the class.

    Parameters
    ----------
    cls : type
        The class to patch
    get_prompt_template : Callable
        A function which takes the same arguments as the class's create method,
        and returns the prompt template to use.
    """
    oldcreate = cls.create

    pii_redactor = Redactor() if redact_pii else None

    def local_create(
        _self,
        *args,
        stream: bool = False,
        libretto: LibrettoCreateParamDict | None = None,
        **kwargs,
    ):
        if libretto is None:
            libretto = LibrettoCreateParams()
        else:
            # Don't mutate the input dict
            libretto = libretto.copy()

        libretto["prompt_template_name"] = (
            libretto["prompt_template_name"]
            or prompt_template_name
            or os.environ.get("LIBRETTO_TEMPLATE_NAME")
            or libretto["api_name"]  # legacy
        )
        libretto["api_key"] = libretto["api_key"] or api_key or os.environ.get("LIBRETTO_API_KEY")
        libretto["chat_id"] = libretto["chat_id"] or chat_id or os.environ.get("LIBRETTO_CHAT_ID")
        libretto["project_key"] = libretto["project_key"] or os.environ.get("LIBRETTO_PROJECT_KEY")
        libretto["feedback_key"] = libretto["feedback_key"] or str(uuid.uuid4())

        if libretto["project_key"] is None and libretto["api_key"] is None:
            return oldcreate(*args, **kwargs, stream=stream)

        if libretto["prompt_template_name"] is None and not allow_unnamed_prompts:
            return oldcreate(*args, **kwargs, stream=stream)

        model_params = kwargs.copy()
        model_params["modelProvider"] = "openai"

        if stream is not None:
            model_params["stream"] = stream

        if libretto["template_params"] is None:
            libretto["template_params"] = {}

        if "messages" in kwargs:
            messages: List[Any] = kwargs["messages"]
            del model_params["messages"]
            model_params["modelType"] = "chat"
            if hasattr(messages, "template"):
                libretto["template_chat"] = cast(TemplateChat, messages).template
            if hasattr(messages, "params"):
                libretto["template_params"].update(cast(TemplateChat, messages).params)

        if "prompt" in kwargs:
            prompt: str = model_params["prompt"]
            if not isinstance(prompt, str):
                raise Exception("Unexpected prompt: want str")
            del model_params["prompt"]
            model_params["modelType"] = "completion"
            if hasattr(prompt, "template"):
                libretto["template_text"] = cast(TemplateString, prompt).template
            if hasattr(prompt, "params"):
                libretto["template_params"].update(cast(TemplateString, prompt).params)

        if libretto["template_text"] is None and libretto["template_chat"] is None:
            template = get_prompt_template(*args, **kwargs)
            if isinstance(template, str):
                libretto["template_text"] = template
            elif isinstance(template, list):
                libretto["template_chat"] = template

        # Redact PII from template parameters if configured to do so
        if pii_redactor and libretto["template_params"]:
            for name, param in libretto["template_params"].items():
                try:
                    libretto["template_params"][name] = pii_redactor.redact(param)
                except Exception as e:
                    logger.warning(
                        "Failed to redact PII from parameter: key=%s, value=%s, error=%s",
                        name,
                        param,
                        e,
                    )

        with event_session(
            project_key=libretto["project_key"],
            api_key=libretto["api_key"],
            prompt_template_name=libretto["prompt_template_name"],
            model_params=model_params,
            prompt_template_text=libretto["template_text"],
            prompt_template_chat=libretto["template_chat"],
            chat_id=libretto["chat_id"],
            prompt_template_params=libretto["template_params"],
            prompt_event_id=libretto["event_id"],
            parent_event_id=libretto["parent_event_id"],
            feedback_key=libretto["feedback_key"],
        ) as complete_event:
            response = oldcreate(*args, **kwargs, stream=stream)

            return_response, event_response = (
                (
                    get_completion_result(cast(Completion, response), False)
                    if not stream
                    else get_completion_result(cast(Iterable[Completion], response), True)
                )
                if isinstance(cls, Completions)
                else (
                    get_chat_result(cast(ChatCompletion, response), False)
                    if not stream
                    else get_chat_result(cast(Iterable[ChatCompletionChunk], response), True)
                )
            )

            # Can only do this for non-streamed responses right now
            if not stream:
                setattr(return_response, "libretto_feedback_key", libretto["feedback_key"])

            # Redact PII before recording the event
            if pii_redactor and event_response:
                try:
                    event_response = pii_redactor.redact_text(event_response)
                except Exception as e:
                    logger.warning(
                        "Failed to redact PII from response: error=%s",
                        e,
                    )

            complete_event(event_response)

        return return_response

    setattr(cls, "create", types.MethodType(local_create, cls))

    def unpatch():
        setattr(cls, "create", oldcreate)

    return unpatch


def list_extract(response: Completion):
    return response, response.choices[0].text


def stream_extract(responses: Iterable[Completion]) -> Tuple[Iterable[Completion], str]:
    (original_response, consumable_response) = tee(responses)
    accumulated = []
    for response in consumable_response:
        accumulated.append(response.choices[0].text)
    return (original_response, "".join(accumulated))


def get_completion_result(response: Completion | Iterable[Completion], stream: bool):
    if isinstance(response, Completion):
        if stream:
            raise ValueError("Streaming is not supported for single responses")
        return list_extract(response)
    return stream_extract(response)


def stream_extract_chat(
    responses: Iterable[ChatCompletionChunk],
) -> Tuple[Iterable[ChatCompletionChunk], str]:
    (original_response, consumable_response) = tee(responses)
    accumulated = []
    for response in consumable_response:
        if response.choices[0].delta.content is not None:
            accumulated.append(response.choices[0].delta.content)
        if response.choices[0].delta.function_call is not None:
            logger.warning("Streaming a function_call response is not supported yet.")
    return (original_response, "".join(accumulated))


def list_extract_chat(response: ChatCompletion):
    content = get_message_content(response.choices[0].message)
    return response, content


def get_chat_result(response: Iterable[ChatCompletionChunk] | ChatCompletion, stream: bool):
    if isinstance(response, ChatCompletion):
        if stream:
            raise ValueError("Streaming is not supported for single responses")
        return list_extract_chat(response)
    return stream_extract_chat(response)


def get_message_content(message: ChatCompletionMessage):
    if message.function_call:
        return json.dumps({"function_call": message.function_call})
    return message.content


def patch_openai(
    client: openai.Client,
    api_key: Optional[str] = None,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
    allow_unnamed_prompts: bool = False,
    redact_pii: bool = False,
):
    """Patch openai APIs to add logging capabilities.

    Returns a function which may be called to "unpatch" the APIs."""

    def get_completion_prompt(*_args, prompt=None, **_kwargs):
        return prompt

    unpatch_completion = _patch_openai_completions(
        client.completions,
        get_completion_prompt,
        api_key=api_key,
        prompt_template_name=prompt_template_name,
        chat_id=chat_id,
        allow_unnamed_prompts=allow_unnamed_prompts,
        redact_pii=redact_pii,
    )

    def get_chat_prompt(*_args, messages=None, **_kwargs):
        # TODO: What should we be sending? For now we'll just send the last message
        prompt_text = messages
        return prompt_text

    unpatch_chat = _patch_openai_completions(
        client.chat.completions,
        get_chat_prompt,
        api_key=api_key,
        prompt_template_name=prompt_template_name,
        chat_id=chat_id,
        allow_unnamed_prompts=allow_unnamed_prompts,
        redact_pii=redact_pii,
    )

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch


@contextmanager
def patched_openai(
    client: openai.Client,
    api_key: Optional[str] = None,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
    allow_unnamed_prompts: bool = False,
    redact_pii: bool = False,
):
    """Simple context manager to wrap patching openai. Usage:

    ```
    with patched_openai():
        # do stuff
    ```
    """
    unpatch = patch_openai(
        client,
        api_key=api_key,
        prompt_template_name=prompt_template_name,
        chat_id=chat_id,
        allow_unnamed_prompts=allow_unnamed_prompts,
        redact_pii=redact_pii,
    )
    yield
    unpatch()
