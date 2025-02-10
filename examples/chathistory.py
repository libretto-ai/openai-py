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
                    "content": "{system_message_1}",
                },
                {
                    "role": "chat_history",
                    "content": "{prev_messages}",
                },
                {
                    "role": "system",
                    "content": "{system_message_2}",
                },
                {
                    "role": "user",
                    "content": "{coach_question}",
                },
            ],
            {
                "system_message_1": "Welcome to the AI Supervisor! I'm here to help you navigate difficult conversations and provide guidance on how to approach challenging situations. How can I assist you today?",
                "system_message_2": "I'm here to help you navigate difficult conversations and provide guidance on how to approach challenging situations. How can I assist you today?",
                "prev_messages": [
                    {"role": "user", "content": "Why are you always late to meetings?"},
                    {
                        "role": "assistant",
                        "content": "It's important to approach feedback and communication in a constructive and non-confrontational manner. Instead of asking a question that may come off as accusatory or blaming, consider rephrasing it to focus on the impact and finding a solution. For example, you could ask something like, \"I've noticed that you arrive late to our meetings. Is there anything going on that is causing this, or is there something we can do to help you be on time in the future?\" This approach opens up a dialogue in a more positive and collaborative way. How does that sound to you?",
                    },
                ],
                "coach_question": "I hear things have been difficult at home for you, I didn't realize that. Thanks for explaining.",
            },
        ),
        libretto=LibrettoCreateParams(
            prompt_template_name="jake-demo-ai-supervisor", chat_id="chat-2"
        ),
    )
    print(chat_completion)


if __name__ == "__main__":
    main()

    # Try to allow the background thread to finish
    time.sleep(2)
