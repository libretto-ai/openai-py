import time

from libretto_openai.libretto_client import LibrettoOpenAIClient
from libretto_openai import TemplateString, LibrettoCreateParams


def main():
    client = LibrettoOpenAIClient()

    template = "Send a greeting to our new user named {name}"
    params = {"name": "Alec"}

    print("TESTING COMPLETION API")
    completion = client.completions.create(
        model="text-davinci-003",
        prompt=TemplateString(template, params),
        libretto=LibrettoCreateParams(
            prompt_template_name="test-from-apitest-completion",
        ),
        stream=True,
    )
    for c in completion:
        print(c)


if __name__ == "__main__":
    main()
    time.sleep(3)
