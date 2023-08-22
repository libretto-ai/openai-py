"""Top-level package for Imaginary Dev OpenAI wrapper."""

__author__ = """Alec Flett"""
__email__ = "alec@thegp.com"
from importlib.metadata import PackageNotFoundError, version

__version__ = version(__package__) if __package__ else "unknown"
from .client import event_session, send_event
from .patch import patch_openai, patched_openai
from .template import TemplateChat, TemplateString
