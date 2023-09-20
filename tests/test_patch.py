import threading
import uuid
from unittest.mock import ANY, MagicMock, patch

import openai
import pytest

from im_openai.patch import patch_openai


@pytest.fixture()
def mock_send_event():
    with patch("im_openai.client.send_event", autospec=True) as mock:
        mock.send_event_called = threading.Event()
        mock.send_event_called.clear()
        mock.side_effect = lambda *args, **kwargs: mock.send_event_called.set()
        yield mock


@pytest.fixture()
def do_patch_openai():
    unpatch = patch_openai()
    yield
    unpatch()


@pytest.fixture()
def mock_chat():
    with patch("openai.ChatCompletion.create", autospec=True) as mock:
        yield mock


def test_chat_completion(mock_chat, mock_send_event: MagicMock, do_patch_openai):
    template = "Send a greeting to our new user named {name}"
    ip_template_params = {"name": "Alec"}
    prompt_text = template.format(**ip_template_params)

    chat_messages = [{"role": "user", "content": prompt_text}]
    chat_template = [{"role": "user", "content": template}]
    api_key = "alecf-local-playground"
    prompt_template_name = "test-from-apitest-chat"
    event_id = uuid.uuid4()
    parent_event_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        temperature=0.4,
        ip_api_key=api_key,
        ip_prompt_template_name=prompt_template_name,
        ip_template_chat=chat_template,
        ip_template_params=ip_template_params,
        ip_parent_event_id=parent_event_id,
        ip_event_id=event_id,
        ip_chat_id=chat_id,
    )
    mock_send_event.send_event_called.wait(1)
    assert tuple(mock_send_event.call_args_list[0]) == (
        (ANY),  # session
        dict(
            api_key=api_key,
            prompt_template_name="test-from-apitest-chat",
            project_key=None,
            prompt=None,
            prompt_event_id=event_id,
            prompt_template_text=None,
            prompt_template_chat=[
                {"role": "user", "content": "Send a greeting to our new user named {name}"}
            ],
            prompt_params=ip_template_params,
            chat_id=chat_id,
            parent_event_id=parent_event_id,
            model_params={
                "model": "gpt-3.5-turbo",
                "modelProvider": "openai",
                "modelType": "chat",
                "temperature": 0.4,
                "stream": False,
            },
            response=ANY,
            response_time=ANY,
            feedback_key=ANY,
        ),
    )
