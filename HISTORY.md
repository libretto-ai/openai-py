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
