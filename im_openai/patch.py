import asyncio
import os
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, cast

import openai

from .client import event_session
from .template import TemplateChat, TemplateString


def patch_openai_class(
    cls,
    get_prompt_template: Callable,
    get_result: Callable,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
):
    oldcreate = cls.create

    def local_create(
        cls,
        *args,
        ip_project_key=None,
        ip_api_key=None,
        ip_prompt_template_name: str | None = None,
        ip_api_name: str | None = None,
        ip_event_id: str | None = None,
        ip_template_text: str | None = None,
        ip_template_chat: List | None = None,
        ip_template_params=None,
        ip_chat_id: str | None = None,
        ip_parent_event_id: str | None = None,
        **kwargs,
    ):
        if ip_project_key is None:
            ip_project_key = os.environ.get("PROMPT_PROJECT_KEY")
        if ip_api_key is None:
            ip_api_key = os.environ.get("PROMPT_API_KEY")
        if ip_project_key is None and ip_api_key is None:
            return oldcreate(*args, **kwargs)

        model_params = kwargs.copy()
        model_params["modelProvider"] = "openai"
        if "messages" in kwargs:
            messages: List[Any] = kwargs["messages"]
            del model_params["messages"]
            model_params["modelType"] = "chat"

            if ip_template_params is None:
                ip_template_params = {}

            if hasattr(messages, "template"):
                ip_template_chat = cast(TemplateChat, messages).template
            if hasattr(messages, "params"):
                ip_template_params.update(cast(TemplateChat, messages).params)

        if "prompt" in kwargs:
            prompt: str = model_params["prompt"]
            del model_params["prompt"]
            model_params["modelType"] = "completion"
            if hasattr(prompt, "template"):
                ip_template_text = cast(TemplateString, prompt).template
            if ip_template_params is None:
                ip_template_params = {}
            if hasattr(prompt, "params"):
                ip_template_params.update(cast(TemplateString, prompt).params)

        if ip_template_text is None and ip_template_chat is None:
            ip_template = get_prompt_template(*args, **kwargs)
            if isinstance(ip_template, str):
                ip_template_text = ip_template
            elif isinstance(ip_template, list):
                ip_template_chat = ip_template

        with event_session(
            project_key=ip_project_key,
            api_key=ip_api_key,
            prompt_template_name=ip_prompt_template_name or prompt_template_name or ip_api_name,
            model_params=model_params,
            prompt_template_text=ip_template_text,
            prompt_template_chat=ip_template_chat,
            chat_id=ip_chat_id or chat_id,
            prompt_template_params=ip_template_params,
            prompt_event_id=ip_event_id,
            parent_event_id=ip_parent_event_id,
        ) as complete_event:
            response = oldcreate(*args, **kwargs)
            complete_event(get_result(response))
        return response

    oldacreate = cls.acreate

    async def local_create_async(cls, *args, template=None, **kwargs):
        # TODO: record the request and response with the template
        return oldacreate(*args, **kwargs)

    setattr(
        cls,
        "create",
        classmethod(lambda cls, *args, **kwds: local_create(cls, *args, **kwds)),
    )
    setattr(
        cls,
        "acreate",
        classmethod(lambda cls, *args, **kwds: local_create_async(cls, *args, **kwds)),
    )

    def unpatch():
        setattr(cls, "create", oldcreate)
        setattr(cls, "acreate", oldacreate)

    return unpatch


def patch_openai(
    api_key: Optional[str] = None,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
):
    """Patch openai APIs to add logging capabilities.

    Returns a function which may be called to "unpatch" the APIs."""

    def get_completion_prompt(*args, prompt=None, **kwargs):
        return prompt

    unpatch_completion = patch_openai_class(
        openai.Completion, get_completion_prompt, lambda x: x["choices"][0]["text"]
    )

    def get_chat_prompt(*args, messages=None, **kwargs):
        # TODO: What should we be sending? For now we'll just send the last message
        prompt_text = messages
        return prompt_text

    unpatch_chat = patch_openai_class(
        openai.ChatCompletion,
        get_chat_prompt,
        lambda x: x["choices"][0]["message"]["content"],
        prompt_template_name=prompt_template_name,
        chat_id=chat_id,
    )

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch


@contextmanager
def patched_openai(
    api_key: Optional[str] = None,
    prompt_template_name: Optional[str] = None,
    chat_id: Optional[str] = None,
):
    """Simple context manager to wrap patching openai. Usage:

    ```
    with patched_openai():
        # do stuff
    ```
    """
    unpatch = patch_openai(prompt_template_name=prompt_template_name, chat_id=chat_id)
    yield
    unpatch()
