from libretto_openai.template import TemplateChat, TemplateString


def test_template_string():
    template = TemplateString("Hello {name}", {"name": "John"})
    assert template.template == "Hello {name}"
    assert template.params == {"name": "John"}
    assert template == "Hello John"
    assert str(template) == "Hello John"


def test_template_chat():
    template = TemplateChat([{"role": "system", "content": "Hello {name}"}], {"name": "John"})
    assert template.template == [{"role": "system", "content": "Hello {name}"}]
    assert template.params == {"name": "John"}
    assert template == [{"role": "system", "content": "Hello John"}]
    assert str(template) == "[{'role': 'system', 'content': 'Hello John'}]"


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
