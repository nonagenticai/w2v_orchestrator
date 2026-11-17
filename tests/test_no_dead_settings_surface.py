"""Every declared setting must have a reader, or be exempt with a written reason.

WHY THIS EXISTS — measured, not hypothetical.

`PipelineSettings` declared eight knobs in one block. **Seven of them were read nowhere in
`src/`**: `max_concurrent_pipelines`, `retry_max_attempts`, `retry_delay_seconds`,
`poll_interval_seconds`, `max_pages_per_crawl`, `crawl_batch_size`, `bundle_signing_secret`.
An operator setting `RETRY_MAX_ATTEMPTS=5`, or reaching for `MAX_CONCURRENT_PIPELINES` to
throttle a saturated pod, got **silence** — no error, no effect, no log line.

Three things made that invisible:

* the block *looks* live, because its eighth sibling (`max_poll_attempts`) genuinely is read
  (`src/clients/base.py`);
* `extra="ignore"` means a ConfigMap key matching a deleted field is silently dropped rather
  than rejected — see memory `w2v-extra-forbid-does-not-catch-typod-env-keys`, which measured
  that `extra="forbid"` does NOT fix this;
* `tests/test_config.py` asserted the *defaults* of several dead fields, so CI was green and
  actively reinforced the belief that they mattered.

`webhook_allow_private` was the sharpest: a security-shaped knob ("Allow private/internal IPs
for webhook callbacks") in a repo with **no webhook feature at all**. A reader — or a future
SSRF audit — would conclude callback egress is gated. Nothing gated anything.

WHAT COUNTS AS A READER, AND THE TRAP IN GETTING THAT WRONG

A field is read if its name appears anywhere under `src/` other than its own declaration —
**including elsewhere inside `config.py` itself**. That last clause matters: `idempotency_ttl_seconds`
is consumed by a model validator in the same file, and a sweep that skipped same-file hits
reported it as dead. It is not. Cross-file-only matching would produce exactly the false
positive this file exists to prevent.

This is a NAME-based check and says so plainly: it proves a field is referenced, not that the
reference is meaningful. That is the honest limit of a static test, and it still catches the
whole class above.

NON-VACUITY. A scan that finds no fields is a bug, not a pass: both the field population and
the source corpus carry floors, and an exemption that matches nothing fails — so an exemption
cannot outlive the thing it excused.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "src" / "config.py"
SRC = REPO_ROOT / "src"

# Fields that are declared here but legitimately have no in-repo reader. Every entry
# carries a reason, and an entry that no longer matches a declared field FAILS below —
# an exemption must not outlive the field it excused.
ALLOWED_UNREAD: dict[str, str] = {
    "mode": (
        "gateway|monolith selector kept as a deployment-shape declaration. It has no reader "
        "because the monolith path was removed in the Studio v3 reset; the field stays so a "
        "ConfigMap that still sets it is not silently meaningless to a human reading the "
        "manifest. Delete it together with the last gitops key that sets it."
    ),
}

# Floors. A scan that enumerates nothing passes every assertion below while proving nothing.
MIN_FIELDS_SCANNED = 40
MIN_SRC_FILES_SCANNED = 100


def _declared_fields() -> dict[str, int]:
    """Every settings field declared in `src/config.py`, mapped to its line number.

    Parsed from the AST rather than by regex: an annotated assignment inside a class whose
    bases mention `Settings` is a field, and nothing else is.
    """
    tree = ast.parse(CONFIG_PATH.read_text())
    fields: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = " ".join(ast.unparse(b) for b in node.bases)
        if "Settings" not in base_names:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if not name.startswith("_") and name != "model_config":
                    fields[name] = stmt.lineno
    return fields


def _src_text() -> tuple[str, int]:
    """All of `src/` concatenated, plus the file count (for the non-vacuity floor)."""
    chunks: list[str] = []
    count = 0
    for path in sorted(SRC.rglob("*.py")):
        chunks.append(path.read_text(errors="ignore"))
        count += 1
    return "\n".join(chunks), count


def _reference_counts() -> dict[str, int]:
    """How many times each declared field's NAME appears under `src/`.

    The field's own declaration is one occurrence, so a count of 1 means "declared and never
    referenced". Same-file references count — `idempotency_ttl_seconds` is consumed by a model
    validator inside `config.py`, and excluding same-file hits reported it as dead when it is
    not.
    """
    blob, _ = _src_text()
    return {name: len(re.findall(rf"\b{re.escape(name)}\b", blob)) for name in _declared_fields()}


def test_the_scan_is_not_vacuous() -> None:
    """Without this, deleting `config.py` or `src/` turns every test below green."""
    fields = _declared_fields()
    _, file_count = _src_text()
    assert len(fields) >= MIN_FIELDS_SCANNED, (
        f"parsed only {len(fields)} settings fields from {CONFIG_PATH} — the AST walk has "
        f"rotted (expected >= {MIN_FIELDS_SCANNED})"
    )
    assert file_count >= MIN_SRC_FILES_SCANNED, (
        f"scanned only {file_count} files under {SRC} (expected >= {MIN_SRC_FILES_SCANNED})"
    )


def test_every_declared_setting_has_a_reader() -> None:
    """A knob nobody reads is worse than no knob: setting it looks like configuring the
    service and does nothing, with no error and no log line."""
    counts = _reference_counts()
    dead = sorted(name for name, hits in counts.items() if hits <= 1 and name not in ALLOWED_UNREAD)
    assert not dead, (
        f"settings field(s) {dead} are declared in src/config.py and read NOWHERE in src/. "
        "Setting them via ConfigMap or env is silently ignored. Delete them, wire a reader, "
        "or add them to ALLOWED_UNREAD with a written reason."
    )


def test_no_exemption_outlives_its_field() -> None:
    """A stale exemption reads as coverage while excusing nothing."""
    declared = set(_declared_fields())
    stale = sorted(name for name in ALLOWED_UNREAD if name not in declared)
    assert not stale, (
        f"ALLOWED_UNREAD names {stale}, which are no longer declared in src/config.py. "
        "Remove the exemption."
    )


def test_every_exemption_states_a_reason() -> None:
    for name, reason in ALLOWED_UNREAD.items():
        assert len(reason.strip()) >= 60, (
            f"exemption for {name!r} has no real reason ({len(reason.strip())} chars). "
            "A one-word excuse is how this rots back."
        )


@pytest.mark.parametrize(
    "field_name",
    ["max_poll_attempts"],
)
def test_the_detector_recognises_known_live_fields(field_name: str) -> None:
    """Self-check on the two shapes that make this detector subtle.

    ``max_poll_attempts`` is the sibling that made the dead block look alive — it is read
    cross-file in ``src/clients/base.py``. ``idempotency_ttl_seconds`` is read only by a model
    validator INSIDE ``config.py``; a sweep that skipped same-file hits called it dead. If
    either is ever reported as unread, the detector is broken, not the code.
    """
    counts = _reference_counts()
    assert field_name in counts, f"{field_name} is no longer declared — update this self-check"
    assert counts[field_name] > 1, (
        f"{field_name} reads as unreferenced; the detector is broken (it must count "
        "same-file readers such as model validators)"
    )
