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

At startup, before any openai calls, patch the library with the
following code:

```python
from im_openai import patch_openai
patch_openai()
```

Then, set the ip_project_key for each request:

```python
import openai

completion = openai.ChatCompletion.create(
    engine="davinci",
    prompt="This is a test",
    ip_project_key="my_project_key"
)
```

If you're using langchain, you can set the ip_project_key in the langchain llm setup:

```python
llm = OpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model_kwargs={"ip_prompt_project_key": "my_project_key"},
)
```

## Credits

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
