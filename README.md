# Imaginary Dev OpenAI wrapper

[![image](https://img.shields.io/pypi/v/im_openai.svg)](https://pypi.python.org/pypi/im_openai)

[![image](https://img.shields.io/travis/alecf/im_openai.svg)](https://travis-ci.com/alecf/im_openai)

[![Documentation Status](https://readthedocs.org/projects/im-openai/badge/?version=latest)](https://im-openai.readthedocs.io/en/latest/?version=latest)

Wrapper library for openai to send events to the Imaginary Programming
monitor

-   Free software: MIT license
-   Documentation: <https://im-openai.readthedocs.io>.

## Features

-   Patches the openai library to allow user to set an ip_project_key
    for each request
-   Works out of the box with langchain

## Get Started

### OpenAI

At startup, before any openai calls, patch the library with the
following code:

```python
from im_openai import patch_openai
patch_openai()
```

Then, set the `ip_project_key` and `ip_api_name` for each request:

```python
import openai

completion = openai.ChatCompletion.create(
    engine="davinci",
    prompt="Show me an emoji that matches the sport: soccer",
    ip_project_key="emojification",
    ip_api_name="sport-emoji",
    ip_template_params={"sport": "soccer"},
    ip_template_chat=[{"role": "user", "content": "Show me an emoji that matches the sport: {sport}" }]
)
```

### Langchain

For langchain, you can directly patch, or use a context manager before setting up a chain:

Using a context manager: (recommended)

```python

from im_openai.langchain import prompt_watch_tracing

with prompt_watch_tracing("emojification", "sport-emoji"):
    chain = LLMChain(llm=...)
    chain.run("Hello world", inputs={"name": "world"})
```

Patch directly:

```python
from im_openai.langchain import prompt_watch_tracing

old_tracer = enable_prompt_watch_tracing("emojification", "sport-emoji",
    template_chat=[{"role": "user", "content": "Show me an emoji that matches the sport: {sport}" }])
chain = LLMChain(llm=...)
chain.run("Hello world", inputs={"name": "world"})

# optional, if you need to disable tracing later
disable_prompt_watch_tracing(old_tracer)
```

### Additional Parameters

Each of the above APIs accept the same additional parameters. The OpenAI API requires a `ip_` prefix for each parameter.

-   `template_chat` / `ip_template_chat`: The chat template to use for the
    request. This is a list of dictionaries with the following keys:

    -   `role`: The role of the speaker. Either `"system"`, `"user"` or `"ai"`.
    -   `content`: The content of the message. This can be a string or a template string with `{}` placeholders.

    For example:

    ```python
    [
      {"role": "ai", "content": "Hello, I'm {system_name}!"},
      {"role": "user", "content": "Hi {system_name}, I'm {user_name}!"}
    ]
    ```

    To represent an array of chat messages, use the artificial role `"chat_history"` with `content` set to the variable name in substitution format: `[{"role": "chat_history", "content": "{prev_messages}"}}]`

-   `template_text` / `ip_template_text`: The text template to use for
    completion-style requests. This is a string or a template string with `{}`
    placeholders, e.g. `"Hello, {user_name}!"`.
-   `chat_id` / `ip_chat_id`: The UUID of a "chat session" - if the chat API is
    being used in a conversational context, then the same chat id can be
    provided so that the events are grouped together, in order. If not provided,
    this will be left blank.

These parameters are only available in the patched OpenAI client:

-   `ip_template_params`: The parameters to use for template
    strings. This is a dictionary of key-value pairs. **Note**: This value is inferred in the Langchain wrapper.
-   `ip_event_id`: A unique UUID for a specific call. If not provided,
    one will be generated. **Note**: In the langchain wrapper, this value is inferred from the `run_id`.
-   `ip_parent_event_id`: The UUID of the parent event. If not provided,
    one will be generated. **Note**: In the langchain wrapper, this value is inferred from the `parent_run_id`.

## Credits

This package was created with Cookiecutter* and the `audreyr/cookiecutter-pypackage`* project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
