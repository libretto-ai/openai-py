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
