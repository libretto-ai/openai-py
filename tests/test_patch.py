import uuid
from unittest.mock import ANY, MagicMock, patch

import openai
import pytest

from im_openai.patch import patch_openai


@pytest.fixture()
def mock_send_event():
    with patch("im_openai.client.send_event", autospec=True) as mock:
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
    project_key = "alecf-local-playground"
    api_name = "test-from-apitest-chat"
    event_id = uuid.uuid4()
    parent_event_id = uuid.uuid4()
    chat_id = uuid.uuid4()
    openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        temperature=0.4,
        ip_project_key=project_key,
        ip_api_name=api_name,
        ip_template_chat=chat_template,
        ip_template_params=ip_template_params,
        ip_parent_event_id=parent_event_id,
        ip_event_id=event_id,
        ip_chat_id=chat_id,
    )

    assert tuple(mock_send_event.call_args_list[0]) == (
        (),
        dict(
            session=ANY,
            project_key=project_key,
            api_name=api_name,
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
            },
        ),
    )
    assert tuple(mock_send_event.call_args_list[1]) == (
        (),
        dict(
            session=ANY,
            project_key="alecf-local-playground",
            api_name="test-from-apitest-chat",
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
            },
            response=ANY,
            response_time=ANY,
        ),
    )
