#!/usr/bin/env python
import logging
import os
import sys
from typing import Dict, cast

import openai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import im_openai

imlogger = logging.getLogger("im_openai")
imlogger.setLevel(logging.DEBUG)
imlogger.addHandler(logging.StreamHandler())


def main():
    print("TESTING CHAT COMPLETION API")
    unpatch = im_openai.patch_openai(
        api_key="619dd081-2f72-4eb1-9f90-3d3c3772334d",
    )
    template = "Send a greeting to our new user named {name}"
    ip_template_params = {"name": "Alec"}
    prompt_text = template.format(**ip_template_params)

    chat_messages = [{"role": "user", "content": prompt_text}]
    chat_template = [{"role": "user", "content": template}]
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        ip_prompt_template_name="test-from-apitest-chat",
        ip_template_chat=chat_template,
        ip_template_params=ip_template_params,
    )
    print(chat_completion)

    print("TESTING COMPLETION API")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text,
        ip_prompt_template_name="test-from-apitest-completion",
        ip_template_text=template,
        ip_template_params=ip_template_params,
    )
    print(completion)

    print("TESTING CHAT STREAMING API")
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        stream=True,
        messages=chat_messages,
        ip_prompt_template_name="test-from-apitest-chat",
        ip_template_chat=chat_template,
        ip_template_params=ip_template_params,
    )
    for chat_result in chat_completion:
        delta = cast(Dict, chat_result)
        if "content" in delta["choices"][0]["delta"]:
            sys.stdout.write(delta["choices"][0]["delta"]["content"])
    print("")

    unpatch()


if __name__ == "__main__":
    main()
