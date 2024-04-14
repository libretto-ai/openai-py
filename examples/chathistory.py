#!/usr/bin/env python

import logging
import time

from libretto_openai import (
    Client,
    LibrettoConfig,
    LibrettoCreateParams,
    TemplateChat,
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

    print("TESTING CHAT COMPLETION API w/ LIBRETTO CHAT_HISTORY")
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=TemplateChat(
            [
                {
                    "role": "system",
                    "content": "My role is to be the AI Coach Supervisor to help guide the coach. I will receive a question from the coach, and I will guide them on the content and quality of the question.",
                },
                {
                    "role": "chat_history",
                    "content": "{prev_messages}",
                },
                {
                    "role": "user",
                    "content": "{coach_question}",
                },
            ],
            {
                "prev_messages": [
                    {"role": "user", "content": "First User message"},
                    {"role": "assistant", "content": "First response from OpenAI"},
                ],
                "coach_question": "Why are you always late to meetings?",
            },
        ),
        libretto=LibrettoCreateParams(prompt_template_name="weather-report", chat_id="chat-1"),
    )
    print(chat_completion)


if __name__ == "__main__":
    main()
