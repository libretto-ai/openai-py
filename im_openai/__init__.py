"""Top-level package for Imaginary Dev OpenAI wrapper."""

__author__ = """Alec Flett"""
__email__ = "alec@thegp.com"
__version__ = "0.1.0"

from openai import ChatCompletion, Completion


def patch_openai_class(cls):
    oldcreate = cls.create

    def local_create(cls, *args, template=None, **kwargs):
        # TODO: record the request and response with the template
        return oldcreate(*args, **kwargs)

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
    unpatch_chat = patch_openai_class(ChatCompletion)
    unpatch_completion = patch_openai_class(Completion)

    def unpatch():
        unpatch_chat()
        unpatch_completion()

    return unpatch
