"""Top-level package for Imaginary Dev OpenAI wrapper."""

__author__ = """Alec Flett"""
__email__ = 'alec@thegp.com'
__version__ = '0.1.0'

from openai import ChatCompletion as OldChatCompletion, Completion as OldCompletion

class ChatCompletion(OldChatCompletion):
    def __init__(self, *args, template=None, **kwargs):
        """ChatCompletion wrapper that allows for a template to be passed in"""
        super().__init__(*args, **kwargs)
        self.template = template

class Completion(OldCompletion):
    """Completion wrapper that allows for a template to be passed in"""
    def __init__(self, *args, template=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = template

