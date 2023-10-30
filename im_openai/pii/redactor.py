from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine

from im_openai.pii.name_recognizer import NameRecognizer
from im_openai.pii.street_address_recognizer import StreetAddressRecognizer
from im_openai.pii.nlp import NoopNlpEngine


class Redactor:
    def __init__(self, recognizers=None):
        registry = RecognizerRegistry(recognizers=recognizers)
        if not recognizers:
            registry.load_predefined_recognizers()
            registry.add_recognizer(NameRecognizer())
            registry.add_recognizer(StreetAddressRecognizer())
        self.analyzer = AnalyzerEngine(registry=registry, nlp_engine=NoopNlpEngine())
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language="en")
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)  # type: ignore
        return anonymized.text
