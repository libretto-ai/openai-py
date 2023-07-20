import asyncio
import os
import uuid
from typing import Any, Dict
from unittest.mock import ANY, MagicMock, patch

import pytest

try:
    import langchain.chains
except ImportError:
    pytest.skip("langchain not installed", allow_module_level=True)
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.load.dump import dumpd
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    StringPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Generation, HumanMessage, LLMResult, SystemMessage

from im_openai import langchain as langchain_util

project_key = "abc"
api_name = "def"


@pytest.fixture(autouse=True)
def openai_api_key():
    os.environ["OPENAI_API_KEY"] = "abc"
    yield
    del os.environ["OPENAI_API_KEY"]


@pytest.fixture()
def mock_send_event():
    with patch("im_openai.client.send_event", autospec=True) as mock:
        yield mock


@pytest.fixture()
def mock_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture()
def pwc():
    return langchain_util.PromptWatchCallbacks(project_key=project_key, api_name=api_name)


def test_llm_start(pwc: langchain_util.PromptWatchCallbacks, mock_send_event: MagicMock):
    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    chain = LLMChain(
        llm=OpenAI(client=None, model="text-davinci-003", temperature=0.5),
        prompt=prompt,
    )
    run_llm_start(pwc, mock_send_event, run_id, parent_run_id, chain)

    mock_send_event.assert_called_once_with(
        ANY,
        project_key=project_key,
        api_name=api_name,
        prompt_template_text="What is a good name for a company that makes {product}?",
        prompt_template_chat=None,
        prompt_params={"product": "socks"},
        prompt_event_id=None,
        chat_id=None,
        model_params={"modelType": "completion", "temperature": 0.5, "model": "text-davinci-003"},
        response=None,
        response_time=None,
        prompt=None,
        parent_event_id=str(parent_run_id),
    )


def test_chat_model_start(pwc: langchain_util.PromptWatchCallbacks, mock_send_event: MagicMock):
    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()
    template_str = (
        "You are a helpful assistant that translates {input_language} to {output_language}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template_str)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=OpenAI(client=None, model="text-davinci-003", temperature=0.5),
        prompt=template,
    )
    template_args = {
        "input_language": "English",
        "output_language": "French",
        "text": "Hello",
    }
    run_chat_model_start(
        pwc,
        mock_send_event,
        run_id,
        parent_run_id,
        chain,
        template_args,
    )

    mock_send_event.assert_called_once_with(
        ANY,
        project_key=project_key,
        api_name=api_name,
        prompt_template_text=None,
        prompt_template_chat=[
            {
                "content": "You are a helpful assistant that translates {input_language} to {output_language}.",
                "role": "system",
            },
            {"content": "{text}", "role": "user"},
        ],
        prompt_params=template_args,
        prompt_event_id=None,
        chat_id=None,
        response=None,
        response_time=None,
        prompt=None,
        model_params={"modelType": "chat", "temperature": 0.5, "model": "text-davinci-003"},
        parent_event_id=str(parent_run_id),
    )


def test_chat_model_template_no_vars(
    pwc: langchain_util.PromptWatchCallbacks, mock_send_event: MagicMock
):
    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()
    template_str = "You are a helpful assistant"
    system_message = SystemMessage(content=template_str)
    human_message = HumanMessage(content="Hello")
    template = ChatPromptTemplate.from_messages([system_message, human_message])
    chain = LLMChain(
        llm=OpenAI(client=None, model="text-davinci-003", temperature=0.5),
        prompt=template,
    )
    template_args = {}
    run_chat_model_start(
        pwc,
        mock_send_event,
        run_id,
        parent_run_id,
        chain,
        template_args,
    )

    mock_send_event.assert_called_once_with(
        ANY,
        project_key=project_key,
        api_name=api_name,
        prompt_template_text=None,
        prompt_template_chat=None,
        prompt_params=template_args,
        prompt_event_id=None,
        chat_id=None,
        response=None,
        response_time=None,
        # Comes in via the prompt rather than the prompt_template_chat/etc
        prompt={
            "chat": [
                {
                    "content": "You are a helpful assistant",
                    "role": "system",
                },
                {"content": "Hello", "role": "user"},
            ]
        },
        model_params={"modelType": "chat", "temperature": 0.5, "model": "text-davinci-003"},
        parent_event_id=str(parent_run_id),
    )


def test_chat_model_parent(
    pwc: langchain_util.PromptWatchCallbacks,
    mock_send_event: MagicMock,
    mock_event_loop,
):
    new_event_id = uuid.uuid4()
    # new_event_future_id = asyncio.futures.Future()
    # new_event_future_id.set_result(new_event_id)
    mock_send_event.return_value = str(new_event_id)
    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()
    template_str = (
        "You are a helpful assistant that translates {input_language} to {output_language}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template_str)

    template = ChatPromptTemplate.from_messages([system_message_prompt])
    chain = LLMChain(
        llm=OpenAI(client=None, model="text-davinci-003", temperature=0.5),
        prompt=template,
    )
    template_args = {
        "input_language": "English",
        "output_language": "French",
    }
    run_chat_model_start(
        pwc,
        mock_send_event,
        run_id,
        parent_run_id,
        chain,
        template_args,
    )

    mock_send_event.assert_called_once_with(
        ANY,
        project_key=project_key,
        api_name=api_name,
        prompt_template_text=None,
        prompt_template_chat=[
            {
                "role": "system",
                "content": "You are a helpful assistant that translates {input_language} to {output_language}.",
            }
        ],
        prompt_params=template_args,
        prompt_event_id=None,
        chat_id=None,
        response=None,
        response_time=None,
        model_params={"modelType": "chat", "temperature": 0.5, "model": "text-davinci-003"},
        prompt=None,
        parent_event_id=str(parent_run_id),
    )
    mock_send_event.reset_mock()

    pwc.on_llm_end(
        LLMResult(generations=[[Generation(text="hi")]]),
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    mock_send_event.assert_called_once_with(
        ANY,
        project_key=project_key,
        api_name=api_name,
        prompt_template_text=None,
        prompt_template_chat=[
            {
                "role": "system",
                "content": "You are a helpful assistant that translates {input_language} to {output_language}.",
            }
        ],
        prompt_params=template_args,
        prompt_event_id=str(new_event_id),
        model_params={"modelType": "chat", "temperature": 0.5, "model": "text-davinci-003"},
        chat_id=None,
        response="hi",
        response_time=ANY,
        prompt=None,
        parent_event_id=str(parent_run_id),
    )


def run_llm_start(
    pwc: langchain_util.PromptWatchCallbacks,
    mock_send_event: MagicMock,
    run_id: uuid.UUID,
    parent_run_id: uuid.UUID,
    chain: LLMChain,
):
    pwc.on_chain_start(
        dumpd(chain),
        dict(product="socks"),
        run_id=run_id,
        parent_run_id=parent_run_id,
    )

    mock_send_event.assert_not_called()

    pwc.on_llm_start(
        dumpd(chain.llm),
        run_id=run_id,
        parent_run_id=parent_run_id,
        prompts=[chain.prompt.format(product="socks")],
    )


def run_chat_model_start(
    pwc: langchain_util.PromptWatchCallbacks,
    mock_send_event: MagicMock,
    run_id: uuid.UUID,
    parent_run_id: uuid.UUID,
    chain: LLMChain,
    args: Dict[str, Any],
):
    pwc.on_chain_start(
        dumpd(chain),
        args,
        run_id=run_id,
        parent_run_id=parent_run_id,
    )
    assert isinstance(chain.prompt, ChatPromptTemplate)

    mock_send_event.assert_not_called()

    messages = chain.prompt.format_messages(**args)
    # TODO: the type coercion here is a bit hacky
    n = langchain_util.format_chat_template(messages)  # type: ignore
    pwc.on_chat_model_start(
        dumpd(chain.llm),
        messages=[n],  # type: ignore
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=None,
        metadata=None,
    )
