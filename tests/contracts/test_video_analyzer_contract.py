"""Contract guard for the video_analyzer ``VerificationResult`` wire model.

``src/clients/schemas/video_analyzer.py::VideoAnalyzerVerification`` is a HAND
MIRROR of a model that lives in a separately deployed repo. It is deliberately
NOT ``extra="forbid"`` — forbidding extras on a body produced by an independently
released server turns every future additive field into a runtime ValidationError
storm across a live pipeline. The cost of that decision is that a field the
server adds and the mirror never learns about is dropped in SILENCE by pydantic's
default ``extra="ignore"``.

That silence has already cost this product twice:

* ``indeterminate`` (the defect this test was written for). The server excludes
  those criteria from ``criteria_total`` and never counts them as failures, so a
  consumer comparing ``criteria_passed == criteria_total`` read a transcription
  backend being DOWN as a clean pass.
* ``VideoAnalyzerAudioGrid.meter``, declared ``str`` against a server that sends
  ``int``. Every real beat grid failed validation, ``music_select`` caught the
  error and degraded, and every reel silently fell back to even spacing — no run
  was ever cut to the beat.

So the guard is the deliverable, not the one-line field addition: whatever the
server declares, the mirror must declare too, at a compatible type.

Provenance
----------
``verification_result.schema.json`` is a VENDORED SNAPSHOT of
``VerificationResult.model_json_schema()`` from ``video_analyzer_service``
``src/api/schemas.py`` at commit ``e291a10``, rendered with
``json.dumps(..., indent=2, sort_keys=True)``. Refresh it with::

    cd video_analyzer_service && .venv/bin/python -c "\
    import json; from src.api.schemas import VerificationResult; \
    print(json.dumps(VerificationResult.model_json_schema(), indent=2, sort_keys=True))" \
      > ../w2v_orchestrator/tests/contracts/verification_result.schema.json

**Stated limit, honestly:** unlike ``compile_request.schema.json`` — which the
``contract-freshness`` CI job regenerates out of the compiler's promoted image on
every PR — this snapshot has NO freshness job, because the analyzer publishes no
contract artifact and its image is not pulled by this repo's CI. Importing the
server package directly is not an option either: ``video_analyzer_service`` is a
separate distribution and is not a dependency here. So this test proves the
mirror matches the analyzer AS OF THAT COMMIT; it cannot notice a field the
analyzer adds tomorrow. Refreshing the snapshot is part of landing an analyzer
schema change. What it does buy is that the drift cannot survive the refresh —
the moment the file is re-copied, an un-mirrored field fails the build.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.clients.schemas.video_analyzer import VideoAnalyzerVerification
from tests.contracts.schema_walk import as_mapping

_VENDORED_SERVER_SCHEMA = Path(__file__).with_name("verification_result.schema.json")

#: What a schema node with no declared ``type`` means: pydantic emits a bare
#: ``{}`` branch for ``Any``. Treated as compatible with anything, which is what
#: ``Any`` means — the mirror narrowing ``Any`` to ``object`` is deliberate.
_ANY = "any"


def _leaf_types(schema: Mapping[str, object]) -> dict[str, frozenset[str]]:
    """Dotted leaf position -> the JSON types the schema allows there.

    Same walk shape as :func:`tests.contracts.schema_walk.schema_leaves` (list
    nesting COLLAPSED, ``$ref`` expanded, ``null`` branches of an optional
    dropped) but it keeps the ``type`` at each leaf, because the defect class
    this guards is half about MISSING fields and half about a field present at
    the WRONG type (``meter: int`` vs ``str``).
    """
    defs: dict[str, object] = as_mapping(schema.get("$defs")) or {}
    out: dict[str, frozenset[str]] = {}
    _walk(dict(schema), "", defs, frozenset(), out)
    return out


def _walk(
    node: dict[str, object],
    path: str,
    defs: dict[str, object],
    visiting: frozenset[str],
    out: dict[str, frozenset[str]],
) -> None:
    ref = node.get("$ref")
    if isinstance(ref, str):
        _walk_ref(ref, path, defs, visiting, out)
        return

    branches = node.get("anyOf") or node.get("oneOf") or node.get("allOf")
    if isinstance(branches, list):
        _walk_union(branches, path, defs, visiting, out)
        return

    items = as_mapping(node.get("items"))
    if items is not None:
        # Collapsed: an array and its element schema occupy the SAME position.
        _walk(items, path, defs, visiting, out)
        return

    properties = as_mapping(node.get("properties"))
    if properties:
        _walk_properties(properties, path, defs, visiting, out)
        return

    declared = node.get("type")
    _record(path, frozenset({str(declared)}) if declared is not None else frozenset({_ANY}), out)


def _walk_ref(
    ref: str,
    path: str,
    defs: dict[str, object],
    visiting: frozenset[str],
    out: dict[str, frozenset[str]],
) -> None:
    name = ref.removeprefix("#/$defs/")
    target = as_mapping(defs.get(name))
    if name in visiting or target is None:  # cycle, or a dangling ref
        _record(path, frozenset({_ANY}), out)
        return
    _walk(target, path, defs, visiting | {name}, out)


def _walk_union(
    branches: Sequence[object],
    path: str,
    defs: dict[str, object],
    visiting: frozenset[str],
    out: dict[str, frozenset[str]],
) -> None:
    descended = False
    for raw in branches:
        branch = as_mapping(raw)
        # The ``| None`` half of an optional carries no contract information —
        # every field on both sides is optional-with-a-default somewhere, and
        # keeping "null" would make each side's OPTIONALITY, rather than its
        # type, the thing being compared.
        if branch is None or branch.get("type") == "null":
            continue
        _walk(branch, path, defs, visiting, out)
        descended = True
    if not descended:
        _record(path, frozenset({"null"}), out)


def _walk_properties(
    properties: dict[str, object],
    path: str,
    defs: dict[str, object],
    visiting: frozenset[str],
    out: dict[str, frozenset[str]],
) -> None:
    for prop_name, prop_schema in properties.items():
        child = f"{path}.{prop_name}" if path else str(prop_name)
        sub = as_mapping(prop_schema)
        if sub is None:
            _record(child, frozenset({_ANY}), out)
        else:
            _walk(sub, child, defs, visiting, out)


def _record(path: str, types: frozenset[str], out: dict[str, frozenset[str]]) -> None:
    if path:
        out[path] = out.get(path, frozenset()) | types


def _server_types() -> dict[str, frozenset[str]]:
    parsed = json.loads(_VENDORED_SERVER_SCHEMA.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"{_VENDORED_SERVER_SCHEMA} is not a JSON object"
    return _leaf_types(parsed)


def _mirror_types() -> dict[str, frozenset[str]]:
    return _leaf_types(VideoAnalyzerVerification.model_json_schema())


def test_mirror_declares_every_field_the_server_sends() -> None:
    """``server - mine`` must be EMPTY.

    This is the direction ``extra="ignore"`` makes invisible at runtime: a field
    the analyzer sends and the mirror never declares is dropped without a trace,
    which is exactly how ``indeterminate`` turned an infra outage into a clean
    pass. The opposite direction (``mine - server``) is checked separately —
    reading a field nobody sends is a dead signal, not a dropped one.
    """
    missing = sorted(set(_server_types()) - set(_mirror_types()))
    assert not missing, (
        "VideoAnalyzerVerification does not mirror field(s) the analyzer sends: "
        f"{missing}. The model is extra='ignore' by design, so these are DROPPED "
        "silently at runtime — add them to src/clients/schemas/video_analyzer.py."
    )


def test_mirror_invents_no_field_the_server_does_not_send() -> None:
    """``mine - server`` must be EMPTY: a field nobody sends is a dead signal."""
    phantom = sorted(set(_mirror_types()) - set(_server_types()))
    assert not phantom, (
        f"VideoAnalyzerVerification declares field(s) the analyzer never sends: {phantom}. "
        "A consumer reading one gets the default forever."
    )


def test_mirrored_fields_have_compatible_types() -> None:
    """Every shared position must accept the same JSON types.

    The half of this defect class the field-name check cannot see: ``meter`` was
    present on both sides and still broke every run, because the mirror said
    ``str`` where the server sends ``int``. ``Any`` (an empty schema branch) is
    compatible with anything by definition.
    """
    server = _server_types()
    mirror = _mirror_types()
    incompatible = {
        path: (sorted(server[path]), sorted(mirror[path]))
        for path in sorted(set(server) & set(mirror))
        if _ANY not in server[path] and _ANY not in mirror[path] and server[path] != mirror[path]
    }
    assert not incompatible, (
        "type drift between the analyzer and its orchestrator mirror "
        f"(path: (server, mine)): {incompatible}"
    )


def test_indeterminate_survives_deserialisation() -> None:
    """The live payload shape, end to end.

    The schema comparison above is structural; this is the behaviour it protects.
    Before the fix this parsed without error and yielded ``[]`` — which is the
    whole problem, because ``[]`` is indistinguishable from "everything was
    measurable".
    """
    parsed = VideoAnalyzerVerification.model_validate(
        {
            "passed": True,
            "criteria_total": 2,
            "criteria_passed": 2,
            "failures": [],
            "indeterminate": ["transcript_similarity"],
            "transcript_similarity": None,
            "screenshot_matches": [],
        }
    )
    assert parsed.indeterminate == ["transcript_similarity"]
    # The read a consumer actually performs. "All criteria passed" is NOT the
    # same claim as "nothing went unmeasured", and a caller must be able to tell
    # them apart from the parsed model alone.
    assert parsed.criteria_passed == parsed.criteria_total
    assert parsed.indeterminate, "a clean-looking pass hid an unevaluated criterion"


def test_unknown_server_field_is_still_tolerated() -> None:
    """``extra='forbid'`` must NOT be added: a newer analyzer must not 500 us.

    Pinning the deliberate half of the design. The guard against a dropped field
    is the schema comparison above, run at snapshot-refresh time — never a
    stricter model config, which would fail CLOSED in production against a server
    released on its own cadence.
    """
    parsed = VideoAnalyzerVerification.model_validate(
        {
            "passed": False,
            "criteria_total": 1,
            "criteria_passed": 0,
            "a_field_from_a_future_analyzer": {"nested": 1},
        }
    )
    assert parsed.passed is False
