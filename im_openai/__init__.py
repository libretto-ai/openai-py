"""Top-level package for Imaginary Dev OpenAI wrapper."""

__author__ = """Alec Flett"""
__email__ = "alec@thegp.com"
__version__ = "0.1.0"
from .client import event_session, send_event
from .patch import patch_openai
