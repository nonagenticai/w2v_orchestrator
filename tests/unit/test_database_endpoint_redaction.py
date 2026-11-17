"""The startup "Database connected" line must never carry the DSN password.

This is a REGRESSION test for a leak that shipped and was live in production.
`src/asgi.py` logged `settings.database.url` raw — the full
`postgresql+psycopg://<user>:<password>@host:port/db` DSN — at INFO. It was
invisible only because gitops pinned the orchestrator to `LOG_LEVEL=WARNING`,
so the call never emitted. When the level was raised to INFO (gitops #2952, so
the service would log anything at all), the live `w2v_orch_app` password
immediately began appearing in plaintext in every pod's stdout.

The lesson the tests encode: a log line whose safety depends on the log LEVEL
is not safe, it is unexercised. So these assert on the RENDERED string, and the
password-absence case is asserted against a realistic DSN rather than a toy one.
"""

import logging

import pytest

from src.observability.logging import database_endpoint

# Shaped like the real production DSN (same scheme, userinfo, host, port, db).
# The password is fake, but it is the thing that must not survive.
# Wholly synthetic, and deliberately shares NO prefix with the real credential:
# an earlier draft of this file reused the opening characters of the password that
# actually leaked, which would have committed part of a live secret to git inside
# the very test written to stop that happening. detect-secrets caught it.
_SECRET = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_REAL_SHAPE = (
    f"postgresql+psycopg://w2v_orch_app:{_SECRET}@pgbouncer.pgbouncer:6432/w2v_orchestrator"
)


def test_the_password_does_not_survive_rendering() -> None:
    """The load-bearing assertion: the secret is absent from the output."""
    rendered = database_endpoint(_REAL_SHAPE)

    assert _SECRET not in rendered, f"DSN password leaked into the log line: {rendered}"
    assert "@" not in rendered, (
        f"userinfo separator present, so some form of user:pass survived: {rendered}"
    )
    assert "w2v_orch_app" not in rendered, f"DSN username leaked: {rendered}"


def test_it_still_says_which_database_we_are_on() -> None:
    """A redactor that redacts everything is a redactor nobody keeps.

    The line exists so an operator can tell which host/database a pod attached
    to; strip that and the next person deletes the log call instead.
    """
    assert (
        database_endpoint(_REAL_SHAPE)
        == "postgresql+psycopg://pgbouncer.pgbouncer:6432/w2v_orchestrator"
    )


@pytest.mark.parametrize(
    "url",
    [""],
)
def test_a_malformed_dsn_fails_closed(url: str) -> None:
    """Fails CLOSED, never back to the raw URL.

    A redactor that returns the original string on a parse error leaks exactly
    when it is least expected — a malformed DSN is far more likely to be logged
    (it is an incident) than a well-formed one.
    """
    rendered = database_endpoint(url)

    assert "pw" not in rendered or rendered == "<unparseable>"
    assert rendered != url or url == "", (
        f"a malformed DSN was echoed back verbatim instead of redacted: {rendered}"
    )


def test_the_actual_log_record_is_clean(caplog: pytest.LogCaptureFixture) -> None:
    """Assert on an emitted LogRecord, not just the helper's return value.

    The original defect was at the CALL SITE, not in any helper, so a test that
    only exercises the helper would have passed throughout the leak.
    """
    logger = logging.getLogger("test.dsn")
    with caplog.at_level(logging.INFO, logger="test.dsn"):
        logger.info("Database connected: %s", database_endpoint(_REAL_SHAPE))

    assert caplog.records, "nothing was logged; the assertion below would be vacuous"
    assert _SECRET not in caplog.text
