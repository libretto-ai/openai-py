#!/usr/bin/env python

import im_openai
import openai


def main():
    unpatch = im_openai.patch_openai()
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
        template="xyz",
    )
    print(chat_completion)

    unpatch()
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello world. This is an unpatched request."}
        ],
    )
    print(chat_completion)


if __name__ == "__main__":
    main()
