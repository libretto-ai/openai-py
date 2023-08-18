from typing import Any, Dict


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
        # make this behave like UserString so we have direct access to the original string
        self.template = template
        self.params = params

    def format(self):
        return [_format_item(item, self.params) for item in self.template]


def _format_item(item, params):
    if isinstance(item, str):
        return TemplateString(item, params)
    if isinstance(item, list):
        return [_format_item(l, params) for l in item]
    if isinstance(item, tuple):
        return tuple(_format_item(l, params) for l in item)
    if isinstance(item, dict):
        return {_format_item(k, params): _format_item(v, params) for k, v in item.items()}
    return item
