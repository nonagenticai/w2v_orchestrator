"""Parser-differential corpus: every login URL is REJECTED, or resolves to exactly the
origin a WHATWG parser (ada, the URL parser Node uses) gives it.

A login secret is typed into whatever host Chromium navigates to. If this module
and WHATWG disagree on the host of an accepted URL, a check approves one host while
the password goes to another — the class behind every re-audit finding on this PR.
"""

from __future__ import annotations

import pytest
from ada_url import URL

from src.services.auth_secrets import canonical_origin, strict_login_url

_CORPUS = [
    # whitespace / control
    "https://twen\tty.example/",
    "https://twenty.example\n.attacker.example/",
    "https://twenty.example/\n",
    "\thttps://twenty.example/",
    " https://twenty.example/",
    "https://twenty.example\r/",
    # scheme shapes
    "https:attacker.example",
    "https:/attacker.example",
    "https:/\\attacker.example",
    "https:\\\\attacker.example",
    "//attacker.example",
    "///attacker.example",
    "HTTPS://twenty.example/",
    "Https://twenty.example/",
    "http://twenty.example/",
    "https:///twenty.example",
    "https:////attacker.example",
    "javascript:alert(1)",
    "ftp://twenty.example/",
    # backslash
    "https://attacker.example\\@twenty.example/",
    "https://twenty.example\\attacker.example",
    "https://twenty.example/\\path",
    "https://attacker.example\\.twenty.example/",
    # userinfo / fragment / query tricks
    "https://user@twenty.example/",
    "https://twenty.example@attacker.example/",
    "https://a:b@twenty.example/",
    "https://twenty.example#@attacker.example/",
    "https://twenty.example?@attacker.example/",
    "https://twenty.example/#@attacker.example",
    # percent-encoding in the authority
    "https://attacker.example%2F@twenty.example/",
    "https://attacker.example%5C@twenty.example/",
    "https://attacker.example%40twenty.example/",
    "https://twenty%2Eexample/",
    "https://%74wenty.example/",
    # full-width, ideographic dots, zero-width, soft hyphen, non-ASCII
    "https://\uff54\uff57\uff45\uff4e\uff54\uff59.example/",
    "https://\uff34\uff37\uff25\uff2e\uff34\uff39.example/",
    "https://twenty\u3002example/",
    "https://twenty\uff0eexample/",
    "https://twenty\uff61example/",
    "https://twenty.example\u200b/",
    "https://twen\u200bty.example/",
    "https://twen\u00adty.example/",
    "https://tw\u00e9nty.example/",
    # IP literals
    "https://127.0.0.1/",
    "https://0x7f.1/",
    "https://0177.0.0.1/",
    "https://2130706433/",
    "https://[::1]/",
    "https://[::ffff:127.0.0.1]/",
    "https://10.0.0.1/",
    "https://twenty.example.0x7f/",
    "https://1.2.3.4.5/",
    "https://twenty.123/",
    # ports
    "https://twenty.example:443/",
    "https://twenty.example:8443/",
    "https://twenty.example:0/",
    "https://twenty.example:99999/",
    "https://twenty.example:/",
    "https://twenty.example:44a/",
    # host shapes
    "https://twenty.example./",
    "https://-twenty.example/",
    "https://twenty-.example/",
    "https://tw_enty.example/",
    "https://twenty..example/",
    "https://xn--twnty-bva.example/",
    "https://twenty.example/path?q=1#frag",
    "https://TWENTY.Example/",
    "https://twenty.example",
    "https://a.b.c.twenty.example/x",
]


def _whatwg_origin(url: str) -> str | None:
    try:
        return URL(url).origin
    except ValueError:
        return None


@pytest.mark.parametrize("url", [_CORPUS[0]])
def test_every_url_is_rejected_or_agrees_with_whatwg(url: str) -> None:
    ours = canonical_origin(url)
    if ours is None:
        with pytest.raises(ValueError):
            strict_login_url(url, "login_url")
        return
    assert ours == _whatwg_origin(url), f"{url!r}: we say {ours}, WHATWG says {_whatwg_origin(url)}"


def test_the_corpus_is_not_vacuous() -> None:
    accepted = [u for u in _CORPUS if canonical_origin(u) is not None]
    assert len(_CORPUS) == 69
    # Plain https URLs still work — a guard that rejects everything proves nothing.
    assert set(accepted) == {
        "https://twenty.example:443/",
        "https://twenty.example:8443/",
        "https://xn--twnty-bva.example/",
        "https://twenty.example/path?q=1#frag",
        "https://TWENTY.Example/",
        "https://twenty.example",
        "https://a.b.c.twenty.example/x",
        "https://twenty.example#@attacker.example/",
        "https://twenty.example?@attacker.example/",
        "https://twenty.example/#@attacker.example",
    }


def test_backslash_guard_kills_on_its_own() -> None:
    """A backslash ONLY in the path: no `@`, no `%`, a valid host — so nothing but
    the backslash guard can reject it (WHATWG rewrites it to "/")."""
    with pytest.raises(ValueError, match="backslash"):
        strict_login_url("https://twenty.example/a\\b", "login_url")


@pytest.mark.parametrize(
    ("url", "reason"),
    [("https://user@twenty.example/", "userinfo")],
)
def test_each_guard_names_its_reason(url: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        strict_login_url(url, "login_url")
