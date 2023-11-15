import openai
from openai import resources

from .completions import LibrettoCompletions, LibrettoChatCompletions
from .types import LibrettoConfig


class LibrettoChat(resources.Chat):
    completions: LibrettoChatCompletions

    def __init__(self, client: openai.Client, config: LibrettoConfig):
        super().__init__(client)
        self.completions = LibrettoChatCompletions(client, config)


class OpenAIClient(openai.Client):
    config: LibrettoConfig
    completions: LibrettoCompletions
    chat: LibrettoChat

    def __init__(self, *args, libretto: LibrettoConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = libretto or LibrettoConfig()
        self.completions = LibrettoCompletions(self, self.config)
        self.chat = LibrettoChat(self, self.config)
