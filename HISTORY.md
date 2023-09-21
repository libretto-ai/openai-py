======= History =======

## 0.1.0 (2023-06-20)

-   First release on PyPI.

## 0.1.1 (2023-06-23)

-   add TemplateString helper and support for data / params

## 0.1.2 (2023-06-23)

-   add support for original template too

## 0.2.0 (2023-06-26)

-   add explicit support for passing the "prompt template text"

## 0.3.0 (2023-06-28)

-   add support for chat templates (as objects instead of arrays)

## 0.4.0 (2023-06-29)

-   switch event reporting to be async / non-blocking

## 0.4.1 (2023-06-29)

-   add utility for formatting langchain messages

## 0.4.2 (2023-06-29)

-   remove stray breakpoint

## 0.4.3 (2023-06-30)

-   pass along chat_id
-   attempt to auto-convert langchain prompt templates

## 0.4.4 (2023-06-30)

-   remove stray prints

## 0.5.0 (2023-07-06)

-   Add langchain callbacks handlers

## 0.6.0 (2023-07-10)

-   Handle duplicate callbacks, agents, etc

## 0.6.1 (2023-07-12)

-   Fix prompt retrieval in deep chains

## 0.6.2 (2023-07-13)

-   Handle cases where input values are not strings

## 0.6.3 (2023-07-18)

-   Better support for server-generated event ids
    (pre-llm sends event, post-llm re-uses the same id)
-   more tests for different kinds of templates

## 0.6.4

-   include temporary patched version of loads()

## 0.7.0

-   breaking change: move im_openai.langchain_util to im_openai.langchain
-   add support for injecting callbacks into all langchain calls using tracing hooks

## 0.7.1

-   Pass along model params to the server

## 0.7.3

-   add explicit support for api_key

## 0.8.0

-   switch to api_key, pretend project_key isn't even a thing

## 0.8.1

-   Used root parent_run_id in langchain calls
-   Unified langchain run id accounting

## 0.8.2

-   added ability to pass `ip_api_name` into langchain template's `additional_kwargs`, like:
    ```python
    template = TemplateString(
        "Hello, {{name}}!",
        additional_kwargs={"ip_api_name": "my-api"},
    )
    ```

## 0.8.3

-   Added context manager for basic openai calls
-   Better docs

## 0.8.4

-   Switched to load() now that it is in langchain proper
-   Resolved `None` to `{varname}` in templates rather than leaving it out

## 0.9.0

-   Complete rewrite of prompt resolution for chats: better support for agents

## 0.9.1

-   Thread through chat_id

## 0.9.2

-   Fixed typos in docs, clarified using `ip_` parameters
-   Flushed out working `TemplateText` / `TemplateChat` templates

## 0.9.3

-   Simplified requirements a bit

## 0.10.0

-   Switch to Poetry for releases

## 0.10.1

-   Add explicit py.typed

## 0.11.0

-   Breaking change: changed api_name to prompt_template_name

## 0.12.0

-   Made event sending happen on a background thread

## 0.12.1

-   Fixed api key propagaination in patched_openai
-   Tried to account for more edge cases in background thread

## 0.12.2

-   Added naive support for streaming (consumes entire stream during logging)

## 0.12.3

-   Fixes for async (`acreate`)
-   added new paramter, `only_named_prompts`, to `patched_openai` to allow for
    only sending events for prompts that have a name

## 0.12.4

-   Send function_call responses as json

## 0.13.0

-   Automatically add `feedback_key` to all requests, and `ip_feedback_key` to responses

## 0.13.1

-   Updated docs for `send_feedback()`
