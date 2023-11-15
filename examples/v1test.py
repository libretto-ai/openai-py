import time

from libretto_openai.v1 import (
    LibrettoCreateParams,
    LibrettoOpenAIClient,
    LibrettoConfig,
    TemplateChat,
    TemplateString,
)


def main():
    client = LibrettoOpenAIClient(
        libretto=LibrettoConfig(
            redact_pii=True,
        )
    )

    template = "Send a greeting to our new user named {name}"
    params = {"name": "Jason"}

    print("TESTING COMPLETION API")
    completion = client.completions.create(
        model="text-davinci-003",
        prompt=TemplateString(template, params),
        libretto=LibrettoCreateParams(
            prompt_template_name="test-from-apitest-completion",
        ),
    )
    print(completion)

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
        # stream=True,
    )
    # for chunk in chat_completion:
    #     print(chunk)
    print(chat_completion)


if __name__ == "__main__":
    main()
    time.sleep(5)
