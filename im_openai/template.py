class TemplateString(str):
    """Wrapper class for strings that allows us to track parameters."""

    data: str
    params: dict

    def __new__(cls, string: str):
        instance = super().__new__(cls, string)
        # make this behave like UserString so we have direct access to the original string
        instance.data = string
        instance.params = {}
        instance.template = None
        return instance
