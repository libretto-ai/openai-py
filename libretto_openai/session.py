"""Utilities for sending events to Libretto."""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, TypedDict
from openai.types.chat import ChatCompletion

import aiohttp

from .background import ensure_background_thread


logger = logging.getLogger(__name__)


def get_url(api_name: str, environment_name: str) -> str:
    if os.environ.get(environment_name):
        return os.environ[environment_name]
    prefix = os.environ.get("LIBRETTO_API_PREFIX", "https://app.getlibretto.com/api")
    return f"{prefix}/{api_name}"


class SendEventResponse(TypedDict):
    id: str
    """The id of the event that was added on the server."""
    api_name: str
    """The name of the API that was used or created, in case null api name was used."""


async def send_event(
    session: aiohttp.ClientSession,
    api_key: str | None,
    *,
    project_key: str | None = None,
    prompt_template_name: str | None,
    prompt_event_id: str | None = None,
    prompt_template_text: str | None = None,
    prompt_template_chat: List | None = None,
    prompt_params: Dict | None = None,
    chat_id: str | None = None,
    response: str | None = None,
    response_time: float | None = None,
    prompt: Any | None = None,
    chain_id: str | None = None,
    model_params: Dict | None = None,
    feedback_key: str | None = None,
    tools: Any | None = None,
    tool_calls: Any | None = None,
    raw_response: Any | None = None,
) -> SendEventResponse | None:
    """Send an event to Libretto. Returns the id of the event that was added on the server."""
    reporting_url = get_url("event", "LIBRETTO_REPORTING_URL")
    event = {
        "apiName": prompt_template_name,
        "params": {},
        "prompt": prompt or {},
        "promptEventId": prompt_event_id,
        "modelParameters": model_params or {},
    }

    logger.debug("Sending event to %s %s", reporting_url, prompt_template_name)
    if project_key is not None:
        event["projectKey"] = project_key
    if api_key is not None:
        event["apiKey"] = api_key

    if feedback_key:
        event["feedbackKey"] = feedback_key

    if not api_key and not project_key:
        logger.warning("No project key or api key set, not sending event")
        return

    if prompt_template_text is not None:
        if isinstance(prompt_template_text, str):
            # first default template to just the raw text
            event["promptTemplateText"] = prompt_template_text
            if prompt_params is None and getattr(prompt_template_text, "params", None):
                # Can be TemplateString or any other
                prompt_params = prompt_template_text.params  # type: ignore

            # If the original template is available, send it too
            if getattr(prompt_template_text, "template", None):
                event["promptTemplateText"] = prompt_template_text.template  # type: ignore

    elif prompt_template_chat is not None:
        # We're going to assume that if a non-string was passed in, then it
        # was probably a chat template, aka a chat message history
        # TODO: figure out template extraction for chat templates
        event["promptTemplateChat"] = prompt_template_chat

    if prompt_params is not None:
        event["params"] = prompt_params
    if response is not None:
        event["response"] = response
    if response_time is not None:
        event["responseTime"] = response_time
    if chat_id is not None:
        event["chatId"] = chat_id
    if chain_id is not None:
        event["chainId"] = chain_id
    if tools is not None:
        event["tools"] = tools
    if tool_calls is not None:
        # model_dump each element in the list
        event["toolCalls"] = [c.model_dump() for c in tool_calls]
    if raw_response is not None:
        # Only add this for non streaming ones for now
        if isinstance(raw_response, ChatCompletion):
            event["rawResponse"] = raw_response.model_dump()

    result = await session.post(reporting_url, json=event)
    json_result: SendEventResponse = await result.json()
    if result.status > 299:
        logger.debug(
            "Event response: %s for %s: %s", result.status, prompt_template_name, json_result
        )
    return json_result


@contextmanager
def event_session(
    project_key: str | None,
    api_key: str | None,
    prompt_template_name: str | None,
    prompt_template_text: str | None,
    prompt_template_chat: List | None,
    model_params: Dict | None = None,
    chat_id: str | None = None,
    prompt_template_params: dict | None = None,
    prompt_event_id: str | None = None,
    chain_id: str | None = None,
    feedback_key: str | None = None,
    tools: Any | None = None,
):
    """Context manager for sending an event to Templatest

    Note: the response time is measured from the moment the context manager is
    entered, not from the moment the event is sent.

    Usage::

        with event_session(
            project_key=project_key,
            prompt_template_name=prompt_template_name,
            prompt_text=prompt_text) as complete_event:
            response = call_llm_api()
            complete_event(response)

    """
    start = time.time()
    if prompt_event_id is None:
        prompt_event_id = str(uuid.uuid4())

    def complete_event(raw_response: Any, response: str, tool_calls: Any = None):
        response_time = (time.time() - start) * 1000
        send_event_background(
            project_key=project_key,
            api_key=api_key,
            prompt_template_name=prompt_template_name,
            prompt_event_id=prompt_event_id,
            prompt_template_text=prompt_template_text,
            prompt_template_chat=prompt_template_chat,
            prompt_params=prompt_template_params,
            chat_id=chat_id,
            chain_id=chain_id,
            model_params=model_params,
            response=response,
            response_time=response_time,
            feedback_key=feedback_key,
            tools=tools,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )

    yield complete_event


def send_event_background(
    *,
    api_key: str | None,
    project_key: str | None = None,
    prompt_template_name: str | None,
    prompt_event_id: str | None = None,
    prompt_template_text: str | None = None,
    prompt_template_chat: List | None = None,
    prompt_params: Dict | None = None,
    chat_id: str | None = None,
    response: str | None = None,
    response_time: float | None = None,
    prompt: Any | None = None,
    chain_id: str | None = None,
    model_params: Dict | None = None,
    feedback_key: str | None = None,
    tools: Any | None = None,
    tool_calls: Any | None = None,
    raw_response: Any | None = None,
):
    """Send an event on a background thread"""

    with ensure_background_thread() as call_in_background:
        call_in_background(
            send_event,
            project_key=project_key,
            api_key=api_key,
            prompt_template_name=prompt_template_name,
            prompt_event_id=prompt_event_id,
            prompt_template_text=prompt_template_text,
            prompt_template_chat=prompt_template_chat,
            prompt_params=prompt_params,
            chat_id=chat_id,
            response=response,
            response_time=response_time,
            prompt=prompt,
            chain_id=chain_id,
            model_params=model_params,
            feedback_key=feedback_key,
            tools=tools,
            tool_calls=tool_calls,
            raw_response=raw_response,
        )


def send_feedback_background(
    *,
    feedback_key: str,
    api_key: str,
    better_response: str | None = None,
    rating: float | None = None,
):
    """Send feedback on a background thread"""
    with ensure_background_thread() as call_in_background:
        call_in_background(
            send_feedback,
            feedback_key=feedback_key,
            api_key=api_key,
            better_response=better_response,
            rating=rating,
        )


async def send_feedback(
    session: aiohttp.ClientSession,
    *,
    feedback_key: str,
    api_key: str,
    better_response: str | None = None,
    rating: float | None = None,
):
    feedback_url = get_url("feedback", "LIBRETTO_FEEDBACK_URL")
    feedback_call = {
        "feedback_key": feedback_key,
        "apiKey": api_key,
        "rating": rating,
        "better_response": better_response,
    }
    result = await session.post(feedback_url, json=feedback_call)
    json: SendEventResponse = await result.json()
    if result.status > 299:
        logger.debug("Feedback response: %s for %s: %s", result.status, feedback_key, json)
    return json


def update_chain_background(
    *,
    chain_id: str,
    api_key: str,
    result: str | None = None,
):
    with ensure_background_thread() as call_in_background:
        call_in_background(
            update_chain,
            chain_id=chain_id,
            api_key=api_key,
            result=result,
        )


async def update_chain(
    session: aiohttp.ClientSession,
    *,
    chain_id: str,
    api_key: str,
    result: str | None = None,
):
    url = get_url("v1/updateChain", "LIBRETTO_UPDATE_CHAIN_URL")
    body = {
        "id": chain_id,
        "apiKey": api_key,
    }
    if result is not None:
        body["result"] = result
    resp = await session.post(url, json=body)
    json: SendEventResponse = await resp.json()
    if resp.status > 299:
        logger.debug("updateChain response: %s for %s: %s", resp.status, id, json)
    return json
