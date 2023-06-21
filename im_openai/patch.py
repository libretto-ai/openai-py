import os
from typing import Callable

from openai import ChatCompletion, Completion

from .client import event_session


def patch_openai_class(cls, get_result: Callable):
    oldcreate = cls.create

    def local_create(
        cls, *args, ip_project_key=None, ip_event_id=None, ip_template_id=None, **kwargs
    ):
        if ip_project_key is None:
            ip_project_key = os.environ.get("PROMPT_PROJECT_KEY")
        if ip_project_key is None:
            return oldcreate(*args, **kwargs)

        with event_session(ip_project_key, ip_event_id) as complete_event:
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


def patch_openai():
    """Patch openai APIs to add logging capabilities.

    Returns a function which may be called to "unpatch" the APIs."""
    unpatch_chat = patch_openai_class(Completion, lambda x: x["choices"][0]["text"])
    unpatch_completion = patch_openai_class(
        ChatCompletion, lambda x: x["choices"][0]["message"]["content"]
    )

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch
