import pytest
from libretto_openai.template import TemplateChat, TemplateString


def test_template_string():
    cases = [
        ("Hello {name}", {"name": "John"}, "Hello John"),
        (
            'Question: {question}\nAnswer: {"is_valid_question": false}',
            {"question": "How are you?"},
            'Question: How are you?\nAnswer: {"is_valid_question": false}',
        ),
    ]
    for template, params, want in cases:
        t = TemplateString(template, params)
        assert t.template == template
        assert t.params == params
        assert t == want
        assert str(t) == want


def test_template_chat():
    cases = [
        (
            [{"role": "system", "content": "Hello {name}"}],
            {"name": "John"},
            [{"role": "system", "content": "Hello John"}],
        ),
        (
            [
                {
                    "role": "system",
                    "content": 'Question: {question}\nAnswer: {"is_valid_question": false}',
                }
            ],
            {"question": "How are you?"},
            [
                {
                    "role": "system",
                    "content": 'Question: How are you?\nAnswer: {"is_valid_question": false}',
                }
            ],
        ),
    ]
    for template, params, want in cases:
        t = TemplateChat(template, params)
        assert t.template == template
        assert t.params == params
        assert t == want
        assert str(t) == str(want)


def test_template_chat_with_chat_history():
    messages = [
        {
            "role": "system",
            "content": "My role is to be the AI Coach Supervisor",
        },
        {
            "role": "chat_history",
            "content": "{prev_messages} {second_history}",
        },
        {
            "role": "user",
            "content": "{coach_question}",
        },
    ]
    template = (
        TemplateChat(
            messages,
            {
                "prev_messages": [
                    {"role": "user", "content": "First User message"},
                    {"role": "assistant", "content": "First response from OpenAI"},
                ],
                "second_history": [
                    {"role": "user", "content": "Second User message"},
                    {"role": "assistant", "content": "Second response from OpenAI"},
                ],
                "coach_question": "Why are you always late to meetings?",
            },
        ),
    )[0]

    expected_result = "[{'role': 'system', 'content': 'My role is to be the AI Coach Supervisor'}, {'role': 'user', 'content': 'First User message'}, {'role': 'assistant', 'content': 'First response from OpenAI'}, {'role': 'user', 'content': 'Second User message'}, {'role': 'assistant', 'content': 'Second response from OpenAI'}, {'role': 'user', 'content': 'Why are you always late to meetings?'}]"
    assert str(template) == expected_result


def test_template_chat_with_chat_history_raises():
    messages = [
        {
            "role": "system",
            "content": "My role is to be the AI Coach Supervisor",
        },
        {
            "role": "chat_history",
            "content": "Previous messages: {prev_history}",
        },
        {
            "role": "user",
            "content": "{coach_question}",
        },
    ]

    expected_err_message = "Only variables are allowed in the chat_history role."
    with pytest.raises(RuntimeError) as exc_info:
        TemplateChat(
            messages,
            {
                "prev_history": [
                    {"role": "user", "content": "First User message"},
                    {"role": "assistant", "content": "First response from OpenAI"},
                ],
                "coach_question": "Why are you always late to meetings?",
            },
        )
    assert str(exc_info.value) == expected_err_message
