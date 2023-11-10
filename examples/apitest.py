#!/usr/bin/env python
import logging
import os
import sys
from typing import Dict, cast

import openai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from libretto_openai import patch_openai, TemplateChat, TemplateString, LibrettoCreateParams


imlogger = logging.getLogger("libretto_openai")
imlogger.setLevel(logging.DEBUG)
imlogger.addHandler(logging.StreamHandler())


def main():
    client = openai.Client()
    unpatch = patch_openai(client)

    template = "Send a greeting to our new user named {name}"
    params = {"name": "Alec"}

    print("TESTING CHAT COMPLETION API")
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=TemplateChat(
            [{"role": "user", "content": template}],
            params,
        ),
        libretto=LibrettoCreateParams(
            prompt_template_name="test-from-apitest-chat",
        ),
    )
    print(chat_completion)

    # print("TESTING COMPLETION API")
    # completion = client.completions.create(
    #     model="text-davinci-003",
    #     prompt=TemplateString(template, params),
    #     libretto=LibrettoCreateParams(
    #         prompt_template_name="test-from-apitest-completion",
    #     ),
    # )
    # print(completion)

    # print("TESTING CHAT STREAMING API")
    # chat_completion = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     stream=True,
    #     messages=TemplateChat(
    #         [{"role": "user", "content": template}],
    #         params,
    #     ),
    #     libretto=LibrettoCreateParams(
    #         prompt_template_name="test-from-apitest-chat",
    #     ),
    # )
    # for chat_result in chat_completion:
    #     delta = cast(Dict, chat_result)
    #     if "content" in delta["choices"][0]["delta"]:
    #         sys.stdout.write(delta["choices"][0]["delta"]["content"])
    # print("")

    unpatch()

    import time

    time.sleep(3)


if __name__ == "__main__":
    main()
