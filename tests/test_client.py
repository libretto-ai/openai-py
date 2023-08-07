import asyncio
import os
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from im_openai.client import send_event


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setitem(os.environ, "PROMPT_REPORTING_URL", "https://app.imaginary.dev/api/event")


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
@patch("os.environ.get")
async def test_send_event_chat(mock_env_get, mock_post, mock_env):
    # Mocking the environment variable for URL
    mock_env_get.return_value = "https://app.imaginary.dev/api/event"

    # Mocking the response from the aiohttp post call
    mock_response = AsyncMock()
    mock_response.json.return_value = {"id": "test_event_id"}
    mock_post.return_value = mock_response

    # Creating a ClientSession object
    async with aiohttp.ClientSession() as session:
        # Testing the send_event function with required parameters
        api_key = "test_api_key"
        prompt_template_chat = [{"role": "user", "content": "Hello world"}]
        prompt_params = {"param1": "value1", "param2": "value2"}
        chat_id = "test_chat_id"
        result = await send_event(
            session,
            api_key,
            api_name="test_api_name",
            prompt_event_id="test_prompt_event_id",
        )
        prompt_params = {"param1": "value1", "param2": "value2"}
        chat_id = "test_chat_id"
        result = await send_event(
            session,
            api_key,
            api_name="test_api_name",
            prompt_event_id="test_prompt_event_id",
            prompt_template_chat=prompt_template_chat,
            prompt_params=prompt_params,
            chat_id=chat_id,
        )

        # Asserting that the returned event id is as expected
        assert result == "test_event_id"

        assert mock_post.call_count == 2
        mock_post.assert_called_with(
            "https://app.imaginary.dev/api/event",
            json={
                "apiName": "test_api_name",
                "params": {"param1": "value1", "param2": "value2"},
                "prompt": {},
                "promptEventId": "test_prompt_event_id",
                "modelParameters": {},
                "apiKey": "test_api_key",
                "promptTemplateChat": [{"role": "user", "content": "Hello world"}],
                "chatId": "test_chat_id",
            },
        )


@pytest.mark.asyncio
@patch("aiohttp.ClientSession.post", new_callable=AsyncMock)
@patch("os.environ.get")
async def test_send_event_text(mock_env_get, mock_post, mock_env):
    # Mocking the environment variable for URL
    mock_env_get.return_value = "https://app.imaginary.dev/api/event"

    # Mocking the response from the aiohttp post call
    mock_response = AsyncMock()
    mock_response.json.return_value = {"id": "test_event_id"}
    mock_post.return_value = mock_response

    # Creating a ClientSession object
    async with aiohttp.ClientSession() as session:
        # Testing the send_event function with required parameters
        api_key = "test_api_key"
        prompt_template_text = "test_prompt_template_text"
        prompt_params = {"param1": "value1", "param2": "value2"}
        chat_id = "test_chat_id"
        result = await send_event(
            session,
            api_key,
            api_name="test_api_name",
            prompt_event_id="test_prompt_event_id",
        )
        prompt_params = {"param1": "value1", "param2": "value2"}
        chat_id = "test_chat_id"
        result = await send_event(
            session,
            api_key,
            api_name="test_api_name",
            prompt_event_id="test_prompt_event_id",
            prompt_template_text=prompt_template_text,
            prompt_params=prompt_params,
            chat_id=chat_id,
        )

        # Asserting that the returned event id is as expected
        assert result == "test_event_id"

        assert mock_post.call_count == 2
        mock_post.assert_called_with(
            "https://app.imaginary.dev/api/event",
            json={
                "apiName": "test_api_name",
                "params": {"param1": "value1", "param2": "value2"},
                "prompt": {},
                "promptEventId": "test_prompt_event_id",
                "modelParameters": {},
                "apiKey": "test_api_key",
                "promptTemplateText": "test_prompt_template_text",
                "chatId": "test_chat_id",
            },
        )
