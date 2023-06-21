import requests
import aiohttp
from contextlib import contextmanager
import os


def send_event(project_key: str, prompt_event_id: str, response=None):
    PROMPT_REPORTING_URL = os.environ.get(
        "PROMPT_REPORTING_URL", "https://app.imaginary.dev/api/event"
    )
    event = {
        "params": {},
        "prompt": {
            # funcName, funcComment, parameterTypes, returnSchema, isImaginary, serviceParameters
        },
        "projectKey": project_key,
        "promptEventId": prompt_event_id,
    }
    response = requests.post(PROMPT_REPORTING_URL, json=event)
    response.raise_for_status()


@contextmanager
def event_session(project_key: str, prompt_event_id: str):
    """Context manager for sending an event to Imaginary Dev.

    Usage::

        with event_session(project_key, prompt_event_id) as complete_event:
            # do something
            complete_event(response)

    """
    send_event(project_key, prompt_event_id)

    def complete_event(response):
        send_event(project_key, prompt_event_id, response)

    yield complete_event
