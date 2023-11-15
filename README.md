# Libretto OpenAI Client

[![image](https://img.shields.io/pypi/v/libretto_openai.svg)](https://pypi.python.org/pypi/libretto_openai)

A drop-in replacement of `openai.Client` for sending events to Libretto.

## Features

- Provides a patched `openai.Client` that allows for setting Libretto-specific parameters for each request.
- Currently supports the synchronous versions of `completions.create()` and `chat.completions.create()`.

## Get Started

To send events to Libretto, you'll need to create a project. From the project you'll need two things:

1. **API key**: (`api_key`) This is generated for the project and is used to identify the project and environment (dev, staging, prod) that the event is coming from.
2. **Template Name**: (`prompt_template_name`) This uniquely identifies a particular prompt that you are using and allows projects to have multiple prompts. This can be in any format but we recommend using a dash-separated format, e.g. `my-prompt-name`.

**Note:** Prompt template names can be auto-generated if the `allow_unnamed_prompts` configuration option is set (see [below](#configuration)). However, if you rely on auto-generated names, new revisions of the same prompt will show up as different prompt templates in Libretto.

### Usage

You can use the `libretto_openai.Client` anywhere that you're currently using the official `openai.Client`.

When instantiating a `libretto_openai.Client`, you can/should provide any of the existing `openai.Client` parameters in the constructor. Libretto-specific configuration can be provided via an additional `libretto` argument (see below).

To allow our tools to separate the "prompt" from the "prompt parameters", use `TemplateChat` and `TemplateText` to create templates.

Use `TemplateChat` For the ChatCompletion APIs:

```python
from libretto_openai import (
    Client,
    LibrettoConfig,
    LibrettoCreateParams,
    TemplateChat,
)

client = Client(
    api_key="<OpenAI API Key>",
    libretto=LibrettoConfig(
        api_key="<Libretto API Key>",
    ),
)

completion = client.chat.completions.create(
    # Standard OpenAI parameters
    model="gpt-3.5-turbo",
    messages=TemplateChat(
        [{"role": "user", "content": "Show me an emoji that matches the sport: {sport}"}],
        {"sport": "soccer"},
    ),
    libretto=LibrettoCreateParams(
        prompt_template_name="sport-emoji",
    ),
)
```

Use `TemplateText` for the Completion API:

```python
from libretto_openai import (
    Client,
    LibrettoConfig,
    LibrettoCreateParams,
    TemplateChat,
)

client = Client(
    api_key="<OpenAI API Key>",
    libretto=LibrettoConfig(
        api_key="<Libretto API Key>",
    ),
)

completion = client.completions.create(
    # Standard OpenAI parameters
    model="text-davinci-003",
    prompt=TemplateText(
        "Show me an emoji that matches the sport: {sport}",
        {"sport": "soccer"},
    ),
    libretto=LibrettoCreateParams(
        prompt_template_name="sport-emoji",
    ),
)
```

#### Advanced usage

##### Manually passing parameters

While the use of `TemplateText` and `TemplateChat` are preferred, you can optionally specify template data inline when calling the `create()` method:

```python
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",

    # Note we are passing the raw messages object here
    messages=[{"role": "user", "content": "Show me an emoji that matches the sport: soccer"}],

    libretto=LibrettoCreateParams(
        # call configuration
        prompt_template_name="sport-emoji",

        # Here the prompt and parameters are passed seperately
        template_params={"sport": "soccer"},
        template_chat=[
            {"role": "user", "content": "Show me an emoji that matches the sport: {sport}"}
        ],
    ),
)
```

### Configuration

The `libretto` kwarg that's present on the `Client` constructor is a `LibrettoConfig` object with the following options:

- `prompt_template_name`: A default name to associate with prompts. If provided,
  this is the name that will be associated with any `create` call that's made
  **without** a `libretto.prompt_template_name` parameter.
- `allow_unnamed_prompts`: When set to `True`, every prompt will be sent to
  Libretto even if no prompt template name as been provided (either via the
  `prompt_template_name` kwarg or via the `libretto.prompt_template_name` parameter on
  `create`). `False` by default.
- `redact_pii`: When `True`, certain personally identifying information (PII) will be attempted to be redacted before being sent to the Libretto backend. See the `pii` package for details about the types of PII being detected/redacted. `False` by default.

### Additional Create Call Parameters

When calling `create()`, a `libretto` argument should be provided to give Libretto-specific context to the call. The following parameters maybe specified:

- `chat_id`: The id of a "chat session" - if the chat API is
    being used in a conversational context, then the same chat id can be
    provided so that the events are grouped together, in order. If not provided,
    this will be left blank.

- `template_chat`: The chat _template_ to record for chat requests. This is a list of dictionaries with the following keys:

  - `role`: The role of the speaker. Either `"system"`, `"user"` or `"ai"`.
  - `content`: The content of the message. This can be a string or a template string with `{}` placeholders.

  For example:

  ```python
  completion = client.chat.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          template_chat=[
              {"role": "ai", "content": "Hello, I'm {system_name}!"},
              {"role": "user", "content": "Hi {system_name}, I'm {user_name}!"}
          ],
      ),
  )
  ```

  To represent an array of chat messages, use the artificial role `"chat_history"` with `content` set to the variable name in substitution format: `[{"role": "chat_history", "content": "{prev_messages}"}}]`

- `template_text`: The text template to record for completion requests. This is a string or a template string with `{}` placeholders.

  For example:

  ```python
  completion = client.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          template_text="Please welcome the user to {system_name}!",
      ),
  )
  ```

- `template_params`: The parameters to use for template strings. This is a dictionary of key-value pairs.

  For example:

  ```python
  completion = client.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          template_text="Please welcome the user to {system_name}!",
          template_params={"system_name": "Awesome Comics Incorporated"},
      ),
  )
  ```

- `event_id`: A unique UUID for a specific call. If not provided, one will be generated.

  For example:

  ```python
  import uuid

  completion = client.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          event_id=uuid.uuid4(),
      ),
  )
  ```

- `parent_event_id`: The UUID of the parent event. All calls with the same parent id are grouped as a "Run Group".

  For example:

  ```python
  import uuid

  parent_id = uuid.uuid4()
  # First call in the run group
  completion = client.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          parent_event_id=parent_id,
      ),
  )

  # Another call in the same group
  completion = client.completions.create(
      ...,
      libretto=LibrettoCreateParams(
          parent_event_id=parent_id,
      ),
  )
  ```

## Sending Feedback

Sometimes the answer provided by the LLM is not ideal, and your users may be
able to help you find better responses. There are a few common cases:

- You might use the LLM to suggest the title of a news article, but let the
    user edit it. If they change the title, you can send feedback to Libretto
    that the answer was not ideal.
- You might provide a chatbot that answers questions, and the user can rate the
    answers with a thumbs up (good) or thumbs down (bad).

You can send this feedback to Libretto by calling `send_feedback()`. This will
send a feedback event to Libretto about a prompt that was previously called, and
let you review this feedback in the Libretto dashboard. You can use this
feedback to develop new tests and improve your prompts.

```python
from libretto_openai import Client

client = Client()
completion = client.completions.create(...)

# Maybe the user didn't like the answer, so ask them for a better one
better_response = ask_user_for_better_response(completion.choices[0].text)

# If the user provided a better answer, send feedback to Libretto
if better_response !== completion.choices[0].text:
    # feedback key is automatically injected into OpenAI response object as an extra field
    feedback_key = completion.model_extra.get("libretto_feedback_key")
    client.send_feedback(
        feedback_key=feedback_key,
        # Better answer from the user
        better_response=better_response,
        # Rating of existing answer, from 0 to 1
        rating=0.2,
    )

```

Note that feedback can include either `rating`, `better_response`, or both.

Parameters:

- `rating` - a value from 0 (meaning the result was completely wrong) to 1 (meaning the result was correct)
- `better_response` - the better response from the user

## Credits

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- Cookiecutter: <https://github.com/audreyr/cookiecutter>
- `audreyr/cookiecutter-pypackage`: <https://github.com/audreyr/cookiecutter-pypackage>
