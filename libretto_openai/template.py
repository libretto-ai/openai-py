import re
from typing import Any, Dict

EXTRACT_PARAM_RE = r"\{(.+?)\}"
ROLE = "role"
CONTENT = "content"
CHAT_HISTORY = "chat_history"


class TemplateString(str):
    """Wrapper class for strings that allows us to track parameters."""

    template: str
    params: Dict[str, Any]

    def __new__(cls, template: str, params: dict):
        s = template.format_map(params)
        instance = super().__new__(cls, s)
        instance.template = template
        instance.params = params
        return instance


class TemplateChat(list):
    """Wrapper class for lists that allows us to track parameters."""

    template: list
    params: dict

    def __init__(self, template: list, params: dict):
        l = _format_item(template, params)
        super().__init__(l)
        # make this behave like TemplateString so we have direct access to the original string
        self.template = template
        self.params = params

    def format(self):
        return [_format_item(item, self.params) for item in self.template]


def _format_item(item, params):
    if isinstance(item, str):
        return TemplateString(item, params)
    if isinstance(item, list):
        all_items = []
        for l in item:
            # We have a reserved keyword for chat history that we do special handling for
            if is_libretto_chat_history(l):
                all_items.extend(expand_chat_history(l, params))
            else:
                all_items.append(_format_item(l, params))
        return all_items
    if isinstance(item, tuple):
        return tuple(_format_item(l, params) for l in item)
    if isinstance(item, dict):
        return {_format_item(k, params): _format_item(v, params) for k, v in item.items()}
    return item


# Returns true of the role of the item is chat_history
def is_libretto_chat_history(item):
    if isinstance(item, dict):
        return item.get(ROLE) == CHAT_HISTORY
    return False


# Finds the chat_history parameter and returns that param list
def expand_chat_history(item, params: Dict[str, Any]):
    content: str = item.get(CONTENT)
    if not content:
        raise RuntimeError("Expected content for the 'chat_history' role but none was found.")

    # Extract the parameter names since we can have more than that
    all_params = re.findall(EXTRACT_PARAM_RE, content)
    if not all_params:
        raise RuntimeError(
            "Expected at least one parameter in the 'chat_history' role but none was found."
        )

    all_messages = []
    for ep in all_params:
        if ep not in params:
            raise RuntimeError(
                f"Expected parameter '{ep}' to be defined in the parameters, but it was not found."
            )
        all_messages.extend(params[ep])

    return all_messages
