"""
name_recognizer implements a recognizer for Presidio that detects peoples' names.

The algorithm has been adapted from github.com/solvvy/redact-pii with license:

The MIT License (MIT)

Copyright (c) 2016 Solvvy <info@solvvy.com> (https://www.solvvy.com/)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import importlib.resources
import json
import re

from presidio_analyzer import (
    RecognizerResult,
    LocalRecognizer,
)


_GREETING_PATTERN = r"(^|\.\s+)(dear|hi|hello|greetings|hey|hey there)"
_CLOSING_PATTERN = r"(thx|thanks|thank you|regards|best|[a-z]+ly|[a-z]+ regards|all the best|happy [a-z]+ing|take care|have a [a-z]+ (weekend|night|day))"

_GREETING_OR_CLOSING_REGEX = re.compile(
    r"(((" + _GREETING_PATTERN + r")|(" + _CLOSING_PATTERN + r"\s*[,.!]*))[\s-]*)", re.I
)
_GENERIC_NAME_REGEX = re.compile(r"( ?(([A-Z][a-z]+)|([A-Z]\.)))+([,.]|[,.]?$)", re.M)

_ENTITY_TYPE = "PERSON"


class NameRecognizer(LocalRecognizer):
    ENTITIES = [_ENTITY_TYPE]

    DEFAULT_EXPLANATION = "Identified as {} by NameRecognizer"

    _well_known_names_regex = None

    def __init__(self):
        super().__init__(
            supported_entities=self.ENTITIES,
            supported_language="en",
        )

        # Lazy load names.json to construct regex
        if not NameRecognizer._well_known_names_regex:
            with importlib.resources.open_text("im_openai.pii.data", "names.json") as f:
                names = json.load(f)
                NameRecognizer._well_known_names_regex = re.compile(
                    r"\b(\s*)(\s*(" + "|".join(names) + r"))+\b", re.I | re.M
                )

    def load(self) -> None:
        pass

    def build_explanation(self, original_score: float, explanation: str):
        pass

    def analyze(self, text: str, entities, nlp_artifacts=None):
        results = []

        greeting_or_closing_match = _GREETING_OR_CLOSING_REGEX.search(text)
        while greeting_or_closing_match:
            generic_name_match = _GENERIC_NAME_REGEX.search(
                text, pos=greeting_or_closing_match.end()
            )
            if generic_name_match:
                bounds = generic_name_match.span()
                if bounds[0] == greeting_or_closing_match.end():
                    suffix = ""
                    try:
                        suffix = generic_name_match.group(5)
                    except IndexError:
                        pass
                    results.append(
                        RecognizerResult(
                            entity_type=_ENTITY_TYPE,
                            start=bounds[0],
                            end=bounds[1] - len(suffix),
                            score=1.0,
                        )
                    )

            greeting_or_closing_match = _GREETING_OR_CLOSING_REGEX.search(
                text, pos=greeting_or_closing_match.end()
            )

        assert NameRecognizer._well_known_names_regex is not None
        for match in NameRecognizer._well_known_names_regex.finditer(text):
            results.append(
                RecognizerResult(
                    entity_type=_ENTITY_TYPE,
                    start=match.end(1),
                    end=match.end(),
                    score=1.0,
                )
            )

        return results
