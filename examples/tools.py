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

    template = "What's the weather like in {location}?"
    params = {"location": "Chicago"}

    print("TESTING CHAT COMPLETION API w/ TOOLS")
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=TemplateChat(
            [{"role": "user", "content": template}],
            params,
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        libretto=LibrettoCreateParams(
            prompt_template_name="weather-report",
        ),
    )
    print(chat_completion)


if __name__ == "__main__":
    main()

    # Try to allow the background thread to finish
    time.sleep(3)
