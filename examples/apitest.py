#!/usr/bin/env python
import os
import sys

import openai

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import im_openai


def main():
    print("TESTING CHAT COMPLETION API")
    unpatch = im_openai.patch_openai()
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
        ip_project_key="alecf-local-playground",
    )
    print(chat_completion)

    print("TESTING COMPLETION API")
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt="Hello world",
        ip_project_key="alecf-local-playground",
    )
    print(completion)

    unpatch()


if __name__ == "__main__":
    main()
