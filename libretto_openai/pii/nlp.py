from typing import Iterable, Iterator, Tuple, Dict, List

from presidio_analyzer.nlp_engine import NlpEngine, NlpArtifacts


class NoopNlpEngine(NlpEngine):
    def __init__(self):
        self.nlp_artifacts = NlpArtifacts([], [], [], [], None, "en")  # type: ignore

    def load(self):
        pass

    def is_loaded(self) -> bool:
        return True

    def is_stopword(self, word, language):
        return False

    def is_punct(self, word, language):
        return False

    def process_text(self, text, language):
        return self.nlp_artifacts

    def process_batch(
        self, texts: Iterable[str], language: str, **kwargs
    ) -> Iterator[Tuple[str, NlpArtifacts]]:
        texts = list(texts)
        for _, txt in enumerate(texts):
            yield txt, self.nlp_artifacts

    def get_nlp_engine_configuration_as_dict(self) -> Dict:
        return {}

    def get_supported_entities(self) -> List[str]:
        return []
