import threading
from typing import List
import uuid
from unittest.mock import ANY, MagicMock, patch

from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionToolParam
from openai.types.chat.chat_completion import Choice
import pytest

from libretto_openai import (
    Client,
    LibrettoConfig,
    LibrettoCreateParams,
    TemplateChat,
)


@pytest.fixture()
def mock_send_event():
    with patch("libretto_openai.session.send_event", autospec=True) as mock:
        mock.send_event_called = threading.Event()
        mock.send_event_called.clear()
        mock.side_effect = lambda *args, **kwargs: mock.send_event_called.set()
        yield mock


def test_chat_completion(mock_send_event: MagicMock):
    client = Client(api_key="test")
    client.chat.completions._original_create = MagicMock()

    template = "Send a greeting to our new user named {name}"
    template_params = {"name": "Alec"}
    prompt_text = template.format(**template_params)

    chat_template = [{"role": "user", "content": template}]
    api_key = "alecf-local-playground"
    prompt_template_name = "test-from-apitest-chat"
    event_id = str(uuid.uuid4())
    chain_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.4,
        libretto=LibrettoCreateParams(
            api_key=api_key,
            prompt_template_name=prompt_template_name,
            template_chat=chat_template,
            template_params=template_params,
            chain_id=chain_id,
            event_id=event_id,
            chat_id=chat_id,
        ),
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
            prompt_params=template_params,
            chat_id=chat_id,
            chain_id=chain_id,
            model_params={
                "model": "gpt-4o-mini",
                "modelProvider": "openai",
                "modelType": "chat",
                "temperature": 0.4,
            },
            response=ANY,
            response_time=ANY,
            feedback_key=ANY,
            tools=None,
            tool_calls=None,
            raw_response=ANY,
        ),
    )


def test_chat_completion_with_tools(mock_send_event: MagicMock):
    client = Client(api_key="test")
    client.chat.completions._original_create = MagicMock()

    template = "Send a greeting to our new user named {name}"
    template_params = {"name": "Alec"}
    prompt_text = template.format(**template_params)
    tools: List[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    chat_template = [{"role": "user", "content": template}]
    api_key = "alecf-local-playground"
    prompt_template_name = "test-from-apitest-chat"
    event_id = str(uuid.uuid4())
    chain_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.4,
        tools=tools,
        libretto=LibrettoCreateParams(
            api_key=api_key,
            prompt_template_name=prompt_template_name,
            template_chat=chat_template,
            template_params=template_params,
            chain_id=chain_id,
            event_id=event_id,
            chat_id=chat_id,
        ),
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
            prompt_params=template_params,
            chat_id=chat_id,
            chain_id=chain_id,
            model_params={
                "model": "gpt-4o-mini",
                "modelProvider": "openai",
                "modelType": "chat",
                "temperature": 0.4,
            },
            response=ANY,
            response_time=ANY,
            feedback_key=ANY,
            tools=tools,
            tool_calls=None,
            raw_response=ANY,
        ),
    )


@pytest.mark.parametrize(
    "redact_pii,expect_params,expect_response",
    (
        [
            True,
            {
                "name": "<PERSON>",
                "body": "here's my card number! <CREDIT_CARD>",
                "history": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "My social is <US_SSN>."},
                ],
            },
            "I'm glad you provided me your private data! Your SSN is <US_SSN>.",
        ],
        [
            False,
            {
                "name": "Alice Johnson",
                "body": "here's my card number! 5105105105105100",
                "history": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "My social is 321-45-6789."},
                ],
            },
            "I'm glad you provided me your private data! Your SSN is 321-45-6789.",
        ],
    ),
)
def test_chat_completion_redact_pii(
    redact_pii, expect_params, expect_response, mock_send_event: MagicMock
):
    mock_chat_create = MagicMock()
    client = Client(
        api_key="test",
        libretto=LibrettoConfig(
            redact_pii=redact_pii,
        ),
    )
    client.chat.completions._original_create = mock_chat_create

    mock_chat_create.return_value = ChatCompletion(
        id="chatcmpl-123",
        object="chat.completion",
        created=1677652288,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="I'm glad you provided me your private data! Your SSN is 321-45-6789.",
                ),
                finish_reason="stop",
            ),
        ],
        usage=CompletionUsage(
            prompt_tokens=9,
            completion_tokens=12,
            total_tokens=21,
        ),
    )

    client.chat.completions.create(
        # Standard OpenAI parameters
        model="gpt-4o-mini",
        messages=TemplateChat(
            [
                {"role": "user", "content": "Send a greeting to our new user named {name}"},
                {
                    "role": "system",
                    "content": "Determine whether a credit card is in this text: {body}",
                },
                {"role": "chat_history", "content": "{history}"},
            ],
            {
                "name": "Alice Johnson",
                "body": "here's my card number! 5105105105105100",
                "history": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "My social is 321-45-6789."},
                ],
            },
        ),
        libretto=LibrettoCreateParams(
            api_key="abc",
            prompt_template_name="test-prompt",
        ),
    )

    mock_send_event.send_event_called.wait(1)
    event_args = mock_send_event.call_args_list[0][1]
    assert event_args["prompt_params"] == expect_params
    assert event_args["response"] == expect_response
