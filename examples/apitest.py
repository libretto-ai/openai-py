#!/usr/bin/env python

import logging
import sys
import time

from libretto_openai import (
    Client,
    LibrettoConfig,
    LibrettoCreateParams,
    TemplateChat,
    TemplateString,
)


imlogger = logging.getLogger("libretto_openai")
imlogger.setLevel(logging.DEBUG)
imlogger.addHandler(logging.StreamHandler())


def main():
    client = Client(
        libretto=LibrettoConfig(
            redact_pii=False,
        )
    )

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

    print("TESTING COMPLETION API")
    completion = client.completions.create(
        model="text-davinci-003",
        prompt=TemplateString(template, params),
        libretto=LibrettoCreateParams(
            prompt_template_name="test-from-apitest-completion",
        ),
    )
    print(completion)

    print("TESTING FEEDBACK")
    if not completion.model_extra or "libretto_feedback_key" not in completion.model_extra:
        raise Exception("Missing libretto_feedback_key")
    client.send_feedback(
        feedback_key=completion.model_extra["libretto_feedback_key"],
        better_response="This response would have been better!",
        rating=0.8,
    )

    print("TESTING CHAT STREAMING API")
    chat_completion_chunks = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=TemplateChat(
            [{"role": "user", "content": template}],
            params,
        ),
        libretto=LibrettoCreateParams(
            prompt_template_name="test-from-apitest-chat",
        ),
        stream=True,
    )
    # Seems like there's a false positive bug in pylint that only occurs when
    # both stream=True and stream=False are used in the same file.
    # pylint: disable=not-an-iterable
    for chunk in chat_completion_chunks:
        if chunk.choices[0].delta.content:
            sys.stdout.write(chunk.choices[0].delta.content)
    print("")


if __name__ == "__main__":
    main()

    # Try to allow the background thread to finish
    time.sleep(3)
