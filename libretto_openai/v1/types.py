from typing import NamedTuple


class LibrettoConfig(NamedTuple):
    api_key: str | None = None
    prompt_template_name: str | None = None
    chat_id: str | None = None
    allow_unnamed_prompts: bool = False
    redact_pii: bool = False
