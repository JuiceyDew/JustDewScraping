"""The pasted-cookie parser.

Pasting the browser's whole `Cookie` header is the primary credential flow, so
the parser has to accept every shape a user might paste and never mangle a token
that itself contains '=' or quotes.
"""

from __future__ import annotations

from ideafindr.web.cookies import parse_cookie_input


def test_header_string():
    assert parse_cookie_input("auth_token=abc; ct0=xyz") == {
        "auth_token": "abc", "ct0": "xyz",
    }


def test_newline_separated():
    assert parse_cookie_input("auth_token=abc\nct0=xyz") == {
        "auth_token": "abc", "ct0": "xyz",
    }


def test_json_object():
    assert parse_cookie_input('{"auth_token": "abc", "ct0": "xyz"}') == {
        "auth_token": "abc", "ct0": "xyz",
    }


def test_netscape_cookies_txt():
    line = ".x.com\tTRUE\t/\tTRUE\t0\tauth_token\tabc"
    assert parse_cookie_input(line) == {"auth_token": "abc"}


def test_base64_value_with_padding_and_equals_survives():
    """Cookie values are opaque; '=' inside a value must not be split on."""
    assert parse_cookie_input("sessionid=abc==; csrftoken=z") == {
        "sessionid": "abc==", "csrftoken": "z",
    }


def test_quoted_values_are_unwrapped():
    assert parse_cookie_input('auth_token="abc"') == {"auth_token": "abc"}


def test_junk_and_blank_never_raise():
    assert parse_cookie_input("") == {}
    assert parse_cookie_input("   \n  ") == {}
    assert parse_cookie_input("not a cookie at all") == {}
    assert parse_cookie_input("=novalue") == {}


def test_comments_are_ignored():
    assert parse_cookie_input("# a comment\nauth_token=abc") == {"auth_token": "abc"}
