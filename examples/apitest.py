#!/usr/bin/env python
import os
import sys

import openai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import im_openai


def main():
    print("TESTING CHAT COMPLETION API")
    unpatch = im_openai.patch_openai()
    template = "Send a greeting to our new user named {name}"
    ip_template_params = {"name": "Alec"}
    prompt_text = template.format(**ip_template_params)

    chat_messages = [{"role": "user", "content": prompt_text}]
    chat_template = [{"role": "user", "content": template}]
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_messages,
        ip_project_key="alecf-local-playground",
        ip_api_name="test-from-apitest-chat",
        ip_template_chat=chat_template,
        ip_template_params=ip_template_params,
    )
    print(chat_completion)

    print("TESTING COMPLETION API")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text,
        ip_project_key="alecf-local-playground",
        ip_api_name="test-from-apitest-completion",
        ip_template_text=template,
        ip_template_params=ip_template_params,
    )
    print(completion)

    unpatch()


if __name__ == "__main__":
    main()