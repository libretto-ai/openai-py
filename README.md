# Libretto OpenAI Wrapper

[![image](https://img.shields.io/pypi/v/libretto_openai.svg)](https://pypi.python.org/pypi/libretto_openai)

Wrapper library for openai to send events to Libretto

## Features

- Patches the openai library to allow user to set Libretto-specific parameters for each request
- Works out of the box with langchain

## Get Started

To send events to Libretto, you'll need to create a project. From the project you'll need two things:

1. **API key**: (`api_key`) This is generated for the project and is used to identify the project and environment (dev, staging, prod) that the event is coming from.
2. **Template Name**: (`prompt_template_name`) This uniquely identifies a particular prompt that you are using and allows projects to have multiple prompts. This can be in any format but we recommend using a dash-separated format, e.g. `my-prompt-name`.

**Note:** Prompt template names can be auto-generated if the `allow_unnamed_prompts` configuration option is set (see [below](#configuration)). However, if you rely on auto-generated names, new revisions of the same prompt will show up as different prompt templates in Libretto.

### OpenAI

You can use the `patched_openai` context manager to patch your code that uses
the existing OpenAI client library:

To allow our tools to separate the "prompt" from the "prompt parameters", use `TemplateChat` and `TemplateText` to create templates.

Use `TemplateChat` For the ChatCompletion APIs:

```python
from libretto_openai import patched_openai, TemplateChat

with patched_openai(api_key="...", prompt_template_name="sport-emoji"):
    import openai

    completion = openai.ChatCompletion.create(
        # Standard OpenAI parameters
        model="gpt-3.5-turbo",
        messages=TemplateChat(
            [{"role": "user", "content": "Show me an emoji that matches the sport: {sport}"}],
            {"sport": "soccer"},
        ),
    )
```

Use `TemplateText` for the Completion API:

```python
from libretto_openai import patched_openai, TemplateText

with patched_openai(api_key="...", prompt_template_name="sport-emoji"):
    import openai

    completion = openai.Completion.create(
        # Standard OpenAI parameters
        model="text-davinci-003",
        prompt=TemplateText("Show me an emoji that matches the sport: {sport}", {"sport": "soccer"}),
    )
```

#### Advanced usage

##### Patching at startup

Rather than using a context manager, you can patch the library once at startup:

```python
from libretto_openai import patch_openai
patch_openai(api_key="...", prompt_template_name="...")
```

Then, you can use the patched library as normal:

```python
import openai

completion = openai.ChatCompletion.create(
    # Standard OpenAI parameters
    ...)
```

##### Manually passing parameters

While the use of `TemplateText` and `TemplateChat` are preferred, most of the parameters passed during patch can also be specified inline when calling the `create()` method.

```python
from libretto_openai import patch_openai, LibrettoCreateParams
patch_openai()

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",

    # Note we are passing the raw chat object here
    messages=[{"role": "user", "content": "Show me an emoji that matches the sport: soccer"}],

    libretto=LibrettoCreateParams(
        # call configuration
        api_key="...",
        prompt_template_name="sport-emoji",

        # Here the prompt and parameters is passed seperately
        template_params={"sport": "soccer"},
        template_chat=[
            {"role": "user", "content": "Show me an emoji that matches the sport: {sport}"}
        ],
    ),
)
```

### Langchain

For langchain, you can directly patch, or use a context manager before setting up a chain:

Using a context manager: (recommended)

```python
from langchain import LLMChain, PromptTemplate, OpenAI
from libretto_openai.langchain import prompt_watch_tracing

with prompt_watch_tracing(api_key="4b2a6608-86cd-4819-aba6-479f9edd8bfb", prompt_template_name="sport-emoji"):
    chain = LLMChain(llm=OpenAI(),
        prompt=PromptTemplate.from_template("What is the capital of {country}?"))
    capital = chain.run({"country": "Sweden"})
```

The `api_key` parameter is visible from your project's settings page.

the prompt_template_name parameter can also be passed directly to a template when you create it, so that it can be tracked separately from other templates:

```python
from langchain import OpenAI, PromptTemplate, LLMChain
from libretto_openai.langchain import prompt_watch_tracing

# The default prompt_template_name is "default-questions"
with prompt_watch_tracing(api_key="4b2a6608-86cd-4819-aba6-479f9edd8bfb", prompt_template_name="default-questions"):
    prompt = PromptTemplate.from_template("""
Please answer the following question: {question}.
""")
    llm = LLMChain(prompt=prompt, llm=OpenAI())
    llm.run(question="What is the meaning of life?")

    # Track user greetings separately under the `user-greeting` api name
    greeting_prompt = PromptTemplate.from_template("""
Please greet our newest forum member, {user}.
Be nice and enthusiastic but not overwhelming.
""",
        additional_kwargs={"libretto_prompt_template_name": "user-greeting"})
    llm = LLMChain(prompt=greeting_prompt, llm=OpenAI(openai_api_key=...))
    llm.run(user="Bob")
```

### Advanced usage

You can patch directly:

```python
from libretto_openai.langchain import enable_prompt_watch_tracing, disable_prompt_watch_tracing

old_tracer = enable_prompt_watch_tracing(api_key="4b2a6608-86cd-4819-aba6-479f9edd8bfb", prompt_template_name="sport-emoji")

prompt = PromptTemplate.from_template("""
Please answer the following question: {question}.
""")
llm = LLMChain(prompt=prompt, llm=OpenAI())
llm.run(question="What is the meaning of life?")

# Track user greetings separately under the `user-greeting` api name
greeting_prompt = PromptTemplate.from_template("""
Please greet our newest forum member, {user}. Be nice and enthusiastic but not overwhelming.
""",
    additional_kwargs={"libretto_prompt_template_name": "user-greeting"})
llm = LLMChain(prompt=greeting_prompt, llm=OpenAI(openai_api_key=...))
llm.run(user="Bob")

# optional, if you need to disable tracing later
disable_prompt_watch_tracing(old_tracer)
```

### Configuration

The following options may be passed as kwargs when patching:

- `prompt_template_name`: A default name to associate with prompts. If provided,
  this is the name that will be associated with any `create` call that's made
  **without** a `libretto.prompt_template_name` parameter.
- `allow_unnamed_prompts`: When set to `True`, every prompt will be sent to
  Libretto even if no prompt template name as been provided (either via the
  `prompt_template_name` kwarg or via the `libretto.prompt_template_name` parameter on
  `create`). `False` by default.
- `redact_pii`: When `True`, certain personally identifying information (PII) will be attempted to be redacted before being sent to the Libretto backend. See the `pii` package for details about the types of PII being detected/redacted. `False` by default.

### Additional Parameters

The following parameters are available in both the patched OpenAI client and the Langchain wrapper.

- For OpenAI, pass these to the `create()` methods.
- For Langchain, pass these to the `prompt_watch_tracing()` context manager or
    the `enable_prompt_watch_tracing()` function.

Parameters:

- `chat_id`: The id of a "chat session" - if the chat API is
    being used in a conversational context, then the same chat id can be
    provided so that the events are grouped together, in order. If not provided,
    this will be left blank.

OpenAI-only parameters:

The following parameters can be specified in the `libretto` object that has been
added to the base OpenAI `create` call interface:

- `template_chat`: The chat _template_ to record for chat requests. This is a list of dictionaries with the following keys:

  - `role`: The role of the speaker. Either `"system"`, `"user"` or `"ai"`.
  - `content`: The content of the message. This can be a string or a template string with `{}` placeholders.

  For example:

  ```python
  completion = openai.ChatCompletion.create(
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
  completion = openai.Completion.create(
      ...,
      libretto=LibrettoCreateParams(
          template_text="Please welcome the user to {system_name}!",
      ),
  )
  ```

- `template_params`: The parameters to use for template strings. This is a dictionary of key-value pairs.

  For example:

  ```python
  completion = openai.Completion.create(
      ...,
      libretto=LibrettoCreateParams(
          template_text="Please welcome the user to {system_name}!",
          template_params={"system_name": "Awesome Comics Incorporated"},
      ),
  )
  ```

- `event_id`: A unique UUID for a specific call. If not provided, one will be generated. **Note**: In the langchain wrapper, this value is inferred from the chain `run_id`.

  For example:

  ```python
  import uuid

  completion = openai.Completion.create(
      ...,
      libretto=LibrettoCreateParams(
          event_id=uuid.uuid4(),
      ),
  )
  ```

- `parent_event_id`: The UUID of the parent event. All calls with the same parent id are grouped as a "Run Group". **Note**: In the langchain wrapper, this value is inferred from the chain `parent_run_id`.

  For example:

  ```python
  import uuid

  parent_id = uuid.uuid4()
  # First call in the run group
  completion = openai.Completion.create(
      ...,
      libretto=LibrettoCreateParams(
          parent_event_id=parent_id,
      ),
  )

  # Another call in the same group
  completion = openai.Completion.create(
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
from libretto_openai import patch_openai, client
patch_openai()

completion = openai.ChatCompletion.create(
    ...)


# Maybe the user didn't like the answer, so ask them for a better one.
better_response = askUserForBetterResult(completion["choices"][0]["text"])

# If the user provided a better answer, send feedback to Libretto
if better_response !== completion["choices"][0]["text"]:
# feedback key is automatically injected into OpenAI response object.
feedback_key = completion.libretto_feedback_key
client.send_feedback(
    api_key=api_key,
    feedback_key=feedback_key,
    # Better answer from the user
    better_response=better_response,
    # Rating of existing answer, from 0 to 1
    rating=0.2)
```

Note that feedback can include either `rating`, `better_response`, or both.

Parameters:

- `rating` - a value from 0 (meaning the result was completely wrong) to 1 (meaning the result was correct)
- `better_response` - the better response from the user

## Credits

This package was created with Cookiecutter and the `audreyr/cookiecutter-pypackage` project template.

- Cookiecutter: <https://github.com/audreyr/cookiecutter>
- `audreyr/cookiecutter-pypackage`: <https://github.com/audreyr/cookiecutter-pypackage>
