from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine

from .name_recognizer import NameRecognizer
from .street_address_recognizer import StreetAddressRecognizer
from .nlp import NoopNlpEngine


class Redactor:
    def __init__(self, recognizers=None):
        registry = RecognizerRegistry(recognizers=recognizers)
        if not recognizers:
            registry.load_predefined_recognizers()
            registry.add_recognizer(NameRecognizer())
            registry.add_recognizer(StreetAddressRecognizer())
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=NoopNlpEngine())
        self.anonymizer = AnonymizerEngine()

    def redact_text(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore
        return anonymized.text

    def redact(self, val):
        if isinstance(val, list):
            return [self.redact(x) for x in val]
        if isinstance(val, tuple):
            return tuple(self.redact(x) for x in val)
        if isinstance(val, dict):
            return {k: self.redact(v) for k, v in val.items()}
        return self.redact_text(str(val))
