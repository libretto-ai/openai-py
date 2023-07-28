import asyncio
import os
from typing import Callable, List

import openai

from .client import event_session


def patch_openai_class(cls, get_prompt_template: Callable, get_result: Callable):
    oldcreate = cls.create

    async def local_create(
        cls,
        *args,
        ip_project_key=None,
        ip_api_key=None,
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
            template_params = {}
            del model_params["messages"]
            model_params["modelType"] = "chat"
            for message in kwargs["messages"]:
                if hasattr(message["content"], "template_args"):
                    template_params.update(message["content"].template_args)
            if ip_template_params is None:
                ip_template_params = {}
            ip_template_params.update(template_params)
        if "prompt" in kwargs:
            del model_params["prompt"]
            model_params["modelType"] = "completion"

        if ip_template_text is None and ip_template_chat is None:
            ip_template = get_prompt_template(*args, **kwargs)
            if isinstance(ip_template, str):
                ip_template_text = ip_template
            elif isinstance(ip_template, list):
                ip_template_chat = ip_template

        async with event_session(
            project_key=ip_project_key,
            api_key=ip_api_key,
            api_name=ip_api_name,
            model_params=model_params,
            prompt_template_text=ip_template_text,
            prompt_template_chat=ip_template_chat,
            chat_id=ip_chat_id,
            prompt_template_params=ip_template_params,
            prompt_event_id=ip_event_id,
            parent_event_id=ip_parent_event_id,
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
        classmethod(lambda cls, *args, **kwds: asyncio.run(local_create(cls, *args, **kwds))),
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
    )

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch
