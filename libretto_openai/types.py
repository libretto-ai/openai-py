from typing import Any, List, NamedTuple, TypedDict


class LibrettoConfig(NamedTuple):
    api_key: str | None = None
    prompt_template_name: str | None = None
    chat_id: str | None = None
    allow_unnamed_prompts: bool = False
    redact_pii: bool = False


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
