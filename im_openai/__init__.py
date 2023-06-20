"""Top-level package for Imaginary Dev OpenAI wrapper."""

__author__ = """Alec Flett"""
__email__ = 'alec@thegp.com'
__version__ = '0.1.0'

from openai import ChatCompletion as OldChatCompletion, Completion as OldCompletion

class ChatCompletion(OldChatCompletion):
    pass

class Completion(OldCompletion):
    pass

