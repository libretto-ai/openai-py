import pytest

from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IpRecognizer,
    PhoneRecognizer,
    UsSsnRecognizer,
    UrlRecognizer,
)

from im_openai.pii import Redactor
from im_openai.pii.name_recognizer import NameRecognizer
from im_openai.pii.street_address_recognizer import StreetAddressRecognizer


@pytest.mark.parametrize(
    "text,expect",
    (
        [
            "Hey it's David Johnson with ACME Corp. Give me a call at 1-510-748-8230",
            "Hey it's <PERSON> with ACME Corp. Give me a call at <PHONE_NUMBER>",
        ],
    ),
)
def test_redactor_pii(text, expect):
    r = Redactor()
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["blah blah\nThis is very important.", "blah blah\nThis is very important."],
        ["blah blah\n\nThank you ..Risemamy McCrubben", "blah blah\n\nThank you ..<PERSON>"],
        ["blah blah. Thanks -Jon", "blah blah. Thanks -<PERSON>"],
        ["here's my Cliff. blah blah", "here's my Cliff. blah blah"],
        ["here's my Clifford. blah blah", "here's my <PERSON>. blah blah"],
        ["Dear Clifford,\n blah blah", "Dear <PERSON>,\n blah blah"],
        [
            "blah blah\n\n\nthanks,\nAnna\n blah blah",
            "blah blah\n\n\nthanks,\n<PERSON>\n blah blah",
        ],
        ["blah blah\n\n\nAnna\n blah blah", "blah blah\n\n\n<PERSON>\n blah blah"],
        ["blah blah\n\n\nAcme Support\n blah blah", "blah blah\n\n\nAcme Support\n blah blah"],
        ["blah blah\n\n\n   Joshua\n blah blah", "blah blah\n\n\n   <PERSON>\n blah blah"],
        [
            "blah blah\n\n\nAll the best,\n\n--Meg C.\n\nAcme Support",
            "blah blah\n\n\nAll the best,\n\n--<PERSON>\n\nAcme Support",
        ],
        [
            "blah blah\n\n\nAll the best,\n\n-John\n\nAcme Support",
            "blah blah\n\n\nAll the best,\n\n-<PERSON>\n\nAcme Support",
        ],
        ["blah blah\nthanks Joshua.\n blah blah", "blah blah\nthanks <PERSON>.\n blah blah"],
        [
            "Hi David Johnson,\nHow are you?\n\nthanks Joshua.\n blah blah",
            "Hi <PERSON>,\nHow are you?\n\nthanks <PERSON>.\n blah blah",
        ],
        ["Subject. Hi David Johnson.", "Subject. Hi <PERSON>."],
        [
            "to hearing from you.\n\nAll the best,\n\nAngel\nCustomer Experience\nwww.foo.com",
            "to hearing from you.\n\nAll the best,\n\n<PERSON>\nCustomer Experience\nwww.foo.com",
        ],
        [
            "getting this sorted out.\n\nKindest regards,\n\nFoo Bar\nCustomer Experience",
            "getting this sorted out.\n\nKindest regards,\n\n<PERSON>\nCustomer Experience",
        ],
        [
            "blah.\n\nAffectionately,\n\nFoo Bar\nblah",
            "blah.\n\nAffectionately,\n\n<PERSON>\nblah",
        ],
        [
            "blah.\n\nHappy Meditating!\n\nFoo Bar\nblah",
            "blah.\n\nHappy Meditating!\n\n<PERSON>\nblah",
        ],
        ["blah.\n\nTake care!\n\nFoo Bar\nblah", "blah.\n\nTake care!\n\n<PERSON>\nblah"],
        [
            "blah.\n\nHave a wonderful weekend.\n\nFoo Bar\nblah",
            "blah.\n\nHave a wonderful weekend.\n\n<PERSON>\nblah",
        ],
        ["blah blah. Thanks -Jon", "blah blah. Thanks -<PERSON>"],
    ),
)
def test_redactor_person(text, expect):
    r = Redactor(recognizers=[NameRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["my VISA card: 4012888888881881.", "my VISA card: <CREDIT_CARD>."],
        ["my MASTERCARD card: 5105105105105100.", "my MASTERCARD card: <CREDIT_CARD>."],
        ["my DISCOVER card: 6011111111111117.", "my DISCOVER card: <CREDIT_CARD>."],
        ["my AMEX card: 3782 822463 10005.", "my AMEX card: <CREDIT_CARD>."],
        ["my AMEX 2nd card: 3782-822463-10005.", "my AMEX 2nd card: <CREDIT_CARD>."],
        ["my AMEX 3rd card: 378282246310005.", "my AMEX 3rd card: <CREDIT_CARD>."],
        ["my DINERS card: 3056 930902 5904.", "my DINERS card: <CREDIT_CARD>."],
        ["my DINERS 2nd card: 3056-930902-5904.", "my DINERS 2nd card: <CREDIT_CARD>."],
        ["my DINERS 3rd card: 30569309025904.", "my DINERS 3rd card: <CREDIT_CARD>."],
    ),
)
def test_redactor_credit_card(text, expect):
    r = Redactor([CreditCardRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["my ssn: 321 45 6789.", "my ssn: <US_SSN>."],
        ["my ssn: 321-45-6789.", "my ssn: <US_SSN>."],
        ["my ssn: 321.45.6789.", "my ssn: <US_SSN>."],
        ["my ssn: 321456789.", "my ssn: <US_SSN>."],
    ),
)
def test_redactor_us_ssn(text, expect):
    r = Redactor([UsSsnRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["my phone: (+44) (555)123-1234.", "my phone: <PHONE_NUMBER>."],
        ["my phone: 1-510-748-8230.", "my phone: <PHONE_NUMBER>."],
        ["my phone: 510.748.8230.", "my phone: <PHONE_NUMBER>."],
        ["my phone: 5107488230.", "my phone: <PHONE_NUMBER>."],
    ),
)
def test_redactor_phone_number(text, expect):
    r = Redactor([PhoneRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["my ip: 10.1.1.235.", "my ip: <IP_ADDRESS>."],
        ["my ip: 2001:0db8:0001:0000:0000:0ab9:C0A8:0102.", "my ip: <IP_ADDRESS>."],
    ),
)
def test_redactor_ip_address(text, expect):
    r = Redactor([IpRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["my email: joe123@solvvy.co.uk.", "my email: <EMAIL_ADDRESS>."],
        ["my email is other+foobar@t.co.", "my email is <EMAIL_ADDRESS>."],
    ),
)
def test_redactor_email(text, expect):
    r = Redactor([EmailRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        ["My homepage is http://example.com", "My homepage is <URL>"],
        [
            "Reset password url is https://example.com/reset/password/12345",
            "Reset password url is <URL>",
        ],
        ["before http://www.example.com after", "before <URL> after"],
        ["before http://www.example.com:123 after", "before <URL>:123 after"],
        ["before http://www.example.com/foo after", "before <URL> after"],
        ["before http://www.example.com/foo/bar after", "before <URL> after"],
        ["before http://www.example.com/foo/bar?foo=bar after", "before <URL> after"],
        ["before http://www.example.com/foo/bar?foo=bar#/foo/bar after", "before <URL> after"],
        [
            "My homepage is http://example.com\nAnd that is that.",
            "My homepage is <URL>\nAnd that is that.",
        ],
    ),
)
def test_redactor_url(text, expect):
    r = Redactor([UrlRecognizer()])
    got = r.redact(text)
    assert got == expect


@pytest.mark.parametrize(
    "text,expect",
    (
        [
            "I live at 123 Park Ave Apt 123 New York City, NY 10002",
            "I live at <STREET_ADDRESS> New York City, NY 10002",
        ],
        [
            "my address is 56 N First St NY 90210",
            "my address is <STREET_ADDRESS> NY 90210",
        ],
    ),
)
def test_redactor_street_address(text, expect):
    r = Redactor([StreetAddressRecognizer()])
    got = r.redact(text)
    assert got == expect
