import os
from typing import Callable

from openai import ChatCompletion, Completion

from .client import event_session
import asyncio


def patch_openai_class(cls, get_prompt_template: Callable, get_result: Callable):
    oldcreate = cls.create

    async def local_create(
        cls,
        *args,
        ip_project_key=None,
        ip_api_name=None,
        ip_event_id=None,
        ip_template_text=None,
        ip_template_chat=None,
        ip_template_params=None,
        **kwargs
    ):
        if ip_project_key is None:
            ip_project_key = os.environ.get("PROMPT_PROJECT_KEY")
        if ip_project_key is None:
            return oldcreate(*args, **kwargs)

        if ip_template_text is None and ip_template_chat is None:
            ip_template = get_prompt_template(*args, **kwargs)
            if isinstance(ip_template, str):
                ip_template_text = ip_template
            elif isinstance(ip_template, list):
                ip_template_chat = ip_template

        async with event_session(
            ip_project_key,
            ip_api_name,
            ip_template_text,
            ip_template_chat,
            ip_template_params,
            ip_event_id,
        ) as complete_event:
            response = oldcreate(*args, **kwargs)
            await complete_event(get_result(response))
        return response

    oldacreate = cls.acreate

    async def local_create_async(cls, *args, template=None, **kwargs):
        # TODO: record the request and response with the template
        return oldacreate(*args, **kwargs)

    setattr(
        cls,
        "create",
        classmethod(
            lambda cls, *args, **kwds: asyncio.run(local_create(cls, *args, **kwds))
        ),
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


def patch_openai():
    """Patch openai APIs to add logging capabilities.

    Returns a function which may be called to "unpatch" the APIs."""

    def get_completion_prompt(*args, prompt=None, **kwargs):
        return prompt

    unpatch_completion = patch_openai_class(
        Completion, get_completion_prompt, lambda x: x["choices"][0]["text"]
    )

    def get_chat_prompt(*args, messages=None, **kwargs):
        # TODO: What should we be sending? For now we'll just send the last message
        prompt_text = messages
        return prompt_text

    unpatch_chat = patch_openai_class(
        ChatCompletion,
        get_chat_prompt,
        lambda x: x["choices"][0]["message"]["content"],
    )

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch