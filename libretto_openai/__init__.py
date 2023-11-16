"""Top-level package for Libretto OpenAI library."""

__author__ = """Alec Flett"""
__email__ = "alec@thegp.com"
from importlib.metadata import PackageNotFoundError, version

__version__ = version(__package__) if __package__ else "unknown"

from .client import Client
from .types import LibrettoConfig, LibrettoCreateParamDict, LibrettoCreateParams
from .template import TemplateChat, TemplateString
