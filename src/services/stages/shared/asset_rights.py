"""Deterministic asset-rights ledger (advisory only) — ORCH-0.7.

Pure helper, no DB, no client, no ``ctx``. Mirrors the frozen-dataclass +
presence-is-opt-in style of ``finish.py``: a total function of the run's
``config_snapshot`` that can never issue a render or a network call.

The ledger enumerates the brand/legal assets a run leans on (the brand name +
logo it stamps on frames, the trademarks + defensibility claims the operator
declared in the ``legal`` brief block) so a reviewer can eyeball what rights the
output asserts. Purely advisory — nothing here gates a run.

Presence-is-opt-in: a snapshot with NEITHER a ``brand`` NOR a ``legal`` block
yields ``[]`` (byte-identical opt-out). Brand rows source from
``config_snapshot["brand"]`` (name/logo_url fed by ``brand_context`` /
``run_setup``); legal rows from ``config_snapshot["legal"]`` (lifted verbatim
from the brief by ``run_setup``). An absent ``legal`` block ⇒ no trademark /
defensibility rows.

A18 phase 2 extends this module with the RECORD half of the defensibility gate:
:class:`AssetLedgerRow` + :func:`build_asset_ledger` widen the walk from the
brand/legal block to EVERY asset the ``config_snapshot`` references (logo, fonts,
music bed, ambient cue, transition + accent SFX, each ingested source URL), and
:func:`trademark_flags` / :func:`defensibility_manifest` turn that walk plus the
claim lint into a content-addressed audit artifact.

Provenance is recorded HONESTLY, never invented. The bundled sound library is the
only asset class whose licence this codebase actually knows: ``sound_catalog``
states every shipped asset is royalty-free for commercial use with no attribution
(``src/services/stages/sound_catalog.py`` module docstring), so bundled catalog ids
get ``license="royalty-free-commercial"`` + ``verified=True``. A TENANT upload (an id
in the reserved ``tenant_`` namespace, P6 ORCH-4) is NOT bundled: a tenant_admin
attested its rights on upload and nothing here can check that claim, so it gets
``license="tenant-declared"`` + ``verified=False`` and is COUNTED in the manifest's
unverified tail — never labelled royalty-free. Nothing else in the
pipeline records a licence — a scraped logo, a site's webfont, an ingested source
page all arrive with no terms attached — so they get ``license="unverified"`` +
``verified=False`` and are COUNTED in the manifest. 100% ledger coverage with a
counted unverified tail is the truthful answer; a fabricated licence string would
not be. Operator-declared trademark / defensibility rows are the one middle case:
the operator asserted the right, so they carry ``license="operator-declared"``.
Nothing here gates a run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TypedDict, cast

from src.services.stages.sound_catalog import (
    DEFAULT_AMBIENT_ID,
    DEFAULT_TRANSITION_ID,
    is_tenant_sound_id,
)

__all__ = [
    "AssetLedgerRow",
    "AssetRightsRow",
    "build_asset_ledger",
    "build_asset_rights_ledger",
    "defensibility_manifest",
    "trademark_flags",
]

#: The only licence this codebase can honestly assert (see the module docstring).
CATALOG_LICENSE = "royalty-free-commercial"
#: Everything else: no licence is recorded anywhere, so we say exactly that.
UNVERIFIED_LICENSE = "unverified"
#: The operator declared this right themselves (legal block).
DECLARED_LICENSE = "operator-declared"
#: A tenant_admin attested the rights to a sound they uploaded (P6 ORCH-4). An
#: assertion, not a licence this codebase can verify.
TENANT_DECLARED_LICENSE = "tenant-declared"

#: Manifest schema version — bump on any shape change (the hash is over the shape).
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class AssetLedgerRow:
    """One asset the finished film rides on, with its recorded rights provenance.

    ``kind`` ∈ {"name", "logo", "font", "music", "ambient", "sfx", "transition",
    "source", "trademark", "defensibility"}. ``ref`` is the asset's identity (a
    catalog id, a URL, a font family, a declared mark). ``source`` is the stage /
    block that PRODUCED it, so a reviewer can trace a row back to the thing that
    put the asset in the plan. ``license`` is always non-empty — an asset whose
    terms nothing in this codebase records says so explicitly
    (:data:`UNVERIFIED_LICENSE`), and ``verified`` is then ``False`` so the
    manifest can count it.
    """

    kind: str
    ref: str
    source: str
    license: str
    attribution: str = ""
    verified: bool = False


@dataclass(frozen=True)
class AssetRightsRow:
    """One asset the run asserts a right to use.

    ``asset_kind`` ∈ {"name", "logo", "trademark", "defensibility"}. ``source``
    records where the row was derived from ("brand_context" for brand-fed rows,
    "legal" for the declared legal block). ``defensible`` marks a legally-asserted
    right (declared trademark / defensibility claim) vs an informational brand
    asset.
    """

    asset_kind: str
    value: str
    source: str = "brand_context"
    defensible: bool = False
    note: str = ""


def _clean(value: object) -> str:
    """Return a non-empty stripped string, else ``""``."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _as_mapping(value: object) -> Mapping[str, object]:
    """An opaque-JSON ``dict`` value as ``Mapping[str, object]``, else an empty map.

    ``isinstance``-narrowing an ``object`` to ``dict`` yields
    ``dict[Unknown, Unknown]`` (Never-typed keys), which rejects string-keyed
    ``.get``; the cast restores the opaque-JSON shape these read-only walks need.
    """
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def build_asset_rights_ledger(config_snapshot: Mapping[str, object]) -> list[AssetRightsRow]:
    """Build the asset-rights ledger from a run's ``config_snapshot``.

    Presence-is-opt-in: a snapshot with neither ``brand`` nor ``legal`` returns
    ``[]``. Rows are emitted in a deterministic order: brand ``name``, brand
    ``logo``, each ``legal.trademarks`` entry (in declared order), then a
    ``legal.defensibility`` row. An absent ``legal`` block ⇒ no trademark /
    defensibility rows (byte-identical opt-out).
    """
    snapshot = config_snapshot if isinstance(config_snapshot, dict) else {}
    brand = _as_mapping(snapshot.get("brand"))
    legal = _as_mapping(snapshot.get("legal"))
    if not brand and not legal:
        return []

    rows: list[AssetRightsRow] = []

    if name := _clean(brand.get("name")):
        rows.append(AssetRightsRow(asset_kind="name", value=name))
    if logo := _clean(brand.get("logo_url")):
        rows.append(AssetRightsRow(asset_kind="logo", value=logo))

    trademarks = legal.get("trademarks")
    if isinstance(trademarks, list):
        for trademark in trademarks:
            if value := _clean(trademark):
                rows.append(
                    AssetRightsRow(
                        asset_kind="trademark",
                        value=value,
                        source="legal",
                        defensible=True,
                        note="operator-declared trademark",
                    )
                )

    if defensibility := _clean(legal.get("defensibility")):
        rows.append(
            AssetRightsRow(
                asset_kind="defensibility",
                value=defensibility,
                source="legal",
                defensible=True,
                note="operator-declared defensibility claim",
            )
        )

    return rows


# ---------------------------------------------------------------- A18 ledger --
def _catalog_row(kind: str, ref: str) -> AssetLedgerRow:
    """A sound-library asset. Bundled → the one licence this codebase truly knows;
    a tenant upload (``tenant_`` id) → tenant-declared and NOT verified."""
    if is_tenant_sound_id(ref):
        return AssetLedgerRow(
            kind=kind,
            ref=ref,
            source="tenant_sound_library",
            license=TENANT_DECLARED_LICENSE,
            attribution="rights attested by the tenant on upload",
            verified=False,
        )
    return AssetLedgerRow(
        kind=kind,
        ref=ref,
        source="sound_catalog",
        license=CATALOG_LICENSE,
        attribution="",  # the bundled library requires none (Mixkit Free / CC0)
        verified=True,
    )


def _unverified_row(kind: str, ref: str, source: str) -> AssetLedgerRow:
    """An asset nothing in this pipeline records terms for. Say so; never invent one."""
    return AssetLedgerRow(
        kind=kind,
        ref=ref,
        source=source,
        license=UNVERIFIED_LICENSE,
        attribution="",
        verified=False,
    )


def _source_ref(entry: object) -> str:
    """The ref of one ``brief.sources`` entry (a bare URL string or a spec dict)."""
    if isinstance(entry, str):
        return _clean(entry)
    if isinstance(entry, dict):
        entry_map = cast("Mapping[str, object]", entry)
        for key in ("url", "ref", "value", "path"):
            if ref := _clean(entry_map.get(key)):
                return ref
    return ""


def _sfx_ref(entry: object) -> str:
    """The catalog id of one ``sound_effects.sfx`` entry (bare id or a cue dict)."""
    if isinstance(entry, str):
        return _clean(entry)
    if isinstance(entry, dict):
        entry_map = cast("Mapping[str, object]", entry)
        for key in ("sfx_id", "id"):
            if ref := _clean(entry_map.get(key)):
                return ref
    return ""


def _sound_rows(sound: Mapping[str, object]) -> list[AssetLedgerRow]:
    """Ledger rows for the resolved sound plan — mirrors ``sound_design``'s resolution.

    Only an ``enabled`` block puts sound in the render, so only an enabled block
    puts sound in the ledger. ``ambient: true`` / an absent ``transitions`` key
    resolve to the same catalog defaults ``sound_design`` resolves them to, so the
    ledger records what will actually PLAY, not what was literally typed.
    """
    if not sound.get("enabled"):
        return []
    rows: list[AssetLedgerRow] = []

    if music := _clean(sound.get("music")):
        rows.append(_catalog_row("music", music))

    ambient = sound.get("ambient")
    if ambient is True:
        rows.append(_catalog_row("ambient", DEFAULT_AMBIENT_ID))
    elif ambient_id := _clean(ambient):
        rows.append(_catalog_row("ambient", ambient_id))

    transitions = sound.get("transitions")
    if transitions is not False:
        transition_id = _clean(transitions) or DEFAULT_TRANSITION_ID
        rows.append(_catalog_row("transition", transition_id))

    sfx = sound.get("sfx")
    if isinstance(sfx, list):
        for ref in sorted({_sfx_ref(entry) for entry in sfx} - {""}):
            rows.append(_catalog_row("sfx", ref))

    return rows


def _brand_rows(brand: Mapping[str, object]) -> list[AssetLedgerRow]:
    """Brand name, logo and fonts. All ``unverified``: ``brand_context`` scrapes the
    source site's identity, and a scrape records no licence for a logo or a webfont."""
    rows: list[AssetLedgerRow] = []
    if name := _clean(brand.get("name")):
        rows.append(_unverified_row("name", name, "brand_context"))
    if logo := _clean(brand.get("logo_url")):
        rows.append(_unverified_row("logo", logo, "brand_context"))
    seen: set[str] = set()
    for key in ("display_font", "body_font", "font_family"):
        font = _clean(brand.get(key))
        if font and font not in seen:
            seen.add(font)
            rows.append(_unverified_row("font", font, "brand_context"))
    return rows


def _declared_rows(config_snapshot: Mapping[str, object]) -> list[AssetLedgerRow]:
    """The operator's own declarations, lifted verbatim from the brief-level walk.

    Reuses :func:`build_asset_rights_ledger` (the ORCH-0.7 brief-level sub-walk)
    rather than re-deriving them, and keeps only its legally-asserted rows — the
    name/logo rows it also emits are already covered by :func:`_brand_rows`.
    """
    return [
        AssetLedgerRow(
            kind=row.asset_kind,
            ref=row.value,
            source="legal",
            license=DECLARED_LICENSE,
            attribution=row.note,
            verified=True,
        )
        for row in build_asset_rights_ledger(config_snapshot)
        if row.defensible
    ]


def build_asset_ledger(config_snapshot: Mapping[str, object]) -> list[AssetLedgerRow]:
    """Every asset the run's ``config_snapshot`` references, with its rights provenance.

    A deterministic, total walk in a fixed order — brand (name → logo → fonts), the
    resolved sound plan (music → ambient → transition → each accent id, sorted),
    each ``brief.sources`` entry in declared order, then the operator's declared
    trademark / defensibility rows. No timestamps, no uuids, no set iteration: the
    same snapshot always yields the same rows, which is what makes the manifest
    hash reproducible.

    Coverage is 100% by construction (AC-7): every row carries a non-empty
    ``license``, and an asset whose terms nothing records is ``unverified`` rather
    than absent or fabricated.
    """
    snapshot = config_snapshot if isinstance(config_snapshot, dict) else {}
    brand = snapshot.get("brand")
    sound = snapshot.get("sound_effects")
    brief = snapshot.get("brief")

    rows: list[AssetLedgerRow] = []
    rows.extend(_brand_rows(_as_mapping(brand)))
    rows.extend(_sound_rows(_as_mapping(sound)))

    sources = _as_mapping(brief).get("sources") if isinstance(brief, dict) else None
    if isinstance(sources, list):
        seen: set[str] = set()
        for entry in sources:
            ref = _source_ref(entry)
            if ref and ref not in seen:
                seen.add(ref)
                rows.append(_unverified_row("source", ref, "source_ingest"))

    rows.extend(_declared_rows(snapshot))
    return rows


# ------------------------------------------------------------ trademark scan --
#: Sentence-initial / common capitalised words that are never a mark.
_COMMON_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "but",
        "for",
        "i",
        "if",
        "in",
        "it",
        "no",
        "not",
        "now",
        "of",
        "on",
        "or",
        "our",
        "so",
        "the",
        "then",
        "this",
        "to",
        "we",
        "why",
        "yes",
        "you",
        "your",
    }
)

#: The word-character class of a mark token. Scraped marketing copy carries BOTH the
#: straight and the typographic apostrophe, so both are word characters here (written
#: as escapes, since the literal glyph is an ambiguous-unicode lint hit).
_WORD_CLASS = r"[\w&.'\u2019-]"

#: A word (which may carry inner dots — ``Node.js``) plus any trailing ™/® marker.
#: Trailing punctuation the word class over-eats (``GitHub.``) is stripped below.
_MARK_RE = re.compile(rf"[A-Za-z]{_WORD_CLASS}*[™®]?", re.UNICODE)
_TOKEN_RE = re.compile(rf"{_WORD_CLASS}+", re.UNICODE)

#: The ™/® markers that make a token a mark wherever it stands.
_MARKERS = "™®"

#: Characters that end a sentence — a capitalised word right after one is just
#: sentence case, not a mark.
_SENTENCE_END = frozenset({".", "!", "?", ":", ";"})

#: Punctuation the word class may swallow at a token's edge.
_EDGE_PUNCT = ".-'\u2019"


def _brand_tokens(brand: Mapping[str, object]) -> frozenset[str]:
    """The first-party mark tokens: every word of the brand name, lower-cased."""
    name = _clean(brand.get("name"))
    return frozenset(token.lower() for token in _TOKEN_RE.findall(name) if token)


def _field(line: object, name: str) -> object:
    """Read ``name`` off a line that may be a ``dict`` or an attribute-bag."""
    if isinstance(line, dict):
        return cast("Mapping[str, object]", line).get(name)
    return getattr(line, name, None)


def _sentence_initial(text: str, start: int) -> bool:
    """Is the token at ``start`` the first word of a sentence?

    True when nothing but whitespace precedes it, or the last non-space character
    before it ends a sentence — in which case its capital is just sentence case.
    """
    prefix = text[:start].rstrip()
    return not prefix or prefix[-1] in _SENTENCE_END


def _is_mark_shaped(word: str) -> bool:
    """CamelCase (an inner capital) or ALLCAPS (≥2 chars) — the shapes a product
    mark takes that ordinary sentence case does not."""
    if len(word) >= 2 and word.isupper() and any(ch.isalpha() for ch in word):
        return True
    return any(ch.isupper() for ch in word[1:])


class TrademarkCandidate(TypedDict):
    """One flagged third-party mark and the line ids that voiced it."""

    mark: str
    line_ids: list[str]


class TrademarkScan(TypedDict):
    """The advisory trademark-scan result: first-party tokens + candidate marks."""

    first_party: list[str]
    third_party_candidates: list[TrademarkCandidate]


def trademark_flags(
    lines: Sequence[object],
    brand: object,
    known_marks: Sequence[str] = (),
) -> TrademarkScan:
    """Scan the spine for third-party marks (AC-9). Advisory — NEVER blocking.

    First-party = the tokens of ``brand["name"]``. A third-party candidate is a
    token that is CamelCase, ALLCAPS (≥2 chars), or carries a ``™``/``®`` marker,
    AND is not sentence-initial (a ™/® marker overrides that: an explicit marker is
    always a mark), not a common word, not a first-party brand token, and not in the
    operator's ``legal.trademark.known_marks`` allow-list.

    Returns ``{"first_party": [...], "third_party_candidates": [{"mark", "line_ids"}]}``,
    everything sorted. False positives are acceptable — this is a review list, not a
    gate. A SILENT inclusion is not: a mark the film says out loud and nobody looked
    at is the failure mode this exists to prevent.
    """
    brand_tokens = _brand_tokens(_as_mapping(brand))
    allowed = {str(mark).strip().lower() for mark in (known_marks or []) if str(mark).strip()}

    hits: dict[str, set[str]] = {}
    for line in lines or []:
        text = str(_field(line, "text") or "")
        line_id = str(_field(line, "id") or "")
        for match in _MARK_RE.finditer(text):
            token = match.group(0)
            marker = token[-1] if token[-1] in _MARKERS else ""
            word = (token[:-1] if marker else token).rstrip(_EDGE_PUNCT)
            if not word:
                continue
            lowered = word.lower()
            if lowered in _COMMON_WORDS or lowered in brand_tokens:
                continue
            if lowered in allowed or f"{lowered}{marker}" in allowed:
                continue
            # An explicit ™/® is a mark wherever it stands. Everything else must be
            # mark-SHAPED (CamelCase / ALLCAPS) and not merely sentence-cased at the
            # start of a sentence.
            if not marker and (_sentence_initial(text, match.start()) or not _is_mark_shaped(word)):
                continue
            hits.setdefault(f"{word}{marker}", set()).add(line_id)

    return TrademarkScan(
        first_party=sorted(brand_tokens),
        third_party_candidates=[
            TrademarkCandidate(mark=mark, line_ids=sorted(line_ids))
            for mark, line_ids in sorted(hits.items())
        ],
    )


# ------------------------------------------------------- defensibility manifest --
def _canonical(payload: Mapping[str, object]) -> str:
    """The canonical byte-string a manifest hashes over: sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def defensibility_manifest(
    *,
    claims: Sequence[Mapping[str, object]],
    ledger: Sequence[AssetLedgerRow],
    disclaimers: Sequence[Mapping[str, object]],
    trademarks: Mapping[str, object],
) -> dict[str, object]:
    """The content-addressed defensibility artifact (AC-10).

    Every claim + its grounding verdict, every asset + its licence, every required
    disclaimer and every third-party mark candidate, hashed into a stable digest.
    The hash is over the canonical JSON of the payload MINUS the hash itself
    (``sort_keys=True``, no separators, no timestamps, no uuid), so identical
    resolved inputs — in any key order — reproduce an identical manifest, and a
    re-render is auditable against the run that produced it.
    """
    rows = sorted(
        (asdict(row) for row in ledger),
        key=lambda row: (row["kind"], row["ref"], row["source"]),
    )
    payload: dict[str, object] = {
        "version": MANIFEST_VERSION,
        "claims": list(claims),
        "ledger": rows,
        "disclaimers": list(disclaimers),
        "trademarks": trademarks,
        "unverified_assets": sum(1 for row in rows if not row["verified"]),
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return {**payload, "hash": digest}
