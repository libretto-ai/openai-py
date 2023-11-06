"""
street_address_recognizer implements a recognizer for Presidio that detects
street addresses.

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

import re
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult


class StreetAddressRecognizer(PatternRecognizer):
    APT_REGEX = r"(apt|bldg|dept|fl|hngr|lot|pier|rm|ste|slip|trlr|unit|#)\.? *[a-z0-9-]+\b"
    ROAD_REGEX = r"(street|st|road|rd|avenue|ave|drive|dr|loop|court|ct|circle|cir|lane|ln|boulevard|blvd|way)\.?\b"
    PO_BOX_REGEX = r"P\.? ?O\.? *Box +\d+"

    STREET_ADDRESS_REGEX = (
        r"(\d+\s*(\w+ ){1,2}" + ROAD_REGEX + r"(\s+" + APT_REGEX + r")?)|(" + PO_BOX_REGEX + r")"
    )

    PATTERNS = [
        Pattern("Street Address", STREET_ADDRESS_REGEX, 1.0),
    ]

    CONTEXT = ["address"]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "STREET_ADDRESS",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

    def analyze(self, *args, **kwargs) -> List[RecognizerResult]:
        # Force case-insensitive
        kwargs["regex_flags"] = kwargs.get("regex_flags", 0) | re.I
        return super().analyze(*args, **kwargs)
