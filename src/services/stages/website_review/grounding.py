"""What the website_review redesign ACTUALLY contains, and a check that the script says only that.

Prod run 01a0a054 (cliffordtraining.nl) narrated "the form sits front and center"
and "security headers are live" over a redesign that has no ``<form>`` anywhere and
— being a static HTML mockup — cannot carry an HTTP header at all. The script prompt
had only the critique findings and the redesign's STYLE name, so the model wrote the
fixes the critique RECOMMENDED as if the redesign had shipped them.

Two halves:

* :func:`build_inventory` — the rendered redesign sections (the ``redesign_section:``
  stills the after act is shown over, in page order) enriched, when the redesign
  HTML can be read, with each section's heading and key elements, plus the set of
  page-level features that are demonstrably present.
* :func:`find_script_violations` — a deterministic post-check of a drafted script
  against that inventory: a feature claim (form, search, chat, cookie banner,
  security headers, HTTPS, …) the redesign does not contain, a BEFORE-act line that
  narrates a fix over the original page, and an AFTER act that is too thin, or
  talks about sections that were never rendered.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

#: The visual role prefix html_render stamps on a per-section redesign still.
SECTION_ROLE_PREFIX: Final = "redesign_section:"

#: Features a static redesign mockup can NEVER demonstrate: they are server/transport
#: configuration, not page content. A line presenting the redesign as having them is
#: ungrounded whatever the HTML says.
SERVER_SIDE_FEATURES: Final = frozenset({"security_headers", "https"})

#: Minimum share of lines the AFTER act must carry on a fully-AI draft.
MIN_AFTER_SHARE: Final = 0.5

# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RedesignSection:
    """One rendered redesign section — the still the after act can put on screen."""

    #: The visual id (``redesign-section-hero``) — the id the visual director maps to.
    id: str
    #: The html_render section name (``hero``, ``diensten``, ``section-2``).
    name: str
    heading: str = ""
    elements: tuple[str, ...] = ()

    @property
    def band_number(self) -> int | None:
        """``N`` for an html_render viewport band (``band-01``), else ``None``.

        html_render slices a redesign with no section landmarks into viewport-tall
        bands named ``band-NN`` (screenshot_renderer). The name is a position, not
        content — printed bare it tells the script model nothing it may say.
        """
        match = _BAND_NAME_RE.fullmatch(self.name)
        return int(match.group(1)) if match else None

    @property
    def has_content(self) -> bool:
        return bool(self.heading or self.elements)

    def describe(self) -> str:
        band = self.band_number
        label = (
            f"screen {band} of the page (no named landmarks)"
            if band is not None and not self.has_content
            else self.name
        )
        parts = [f"section_id {self.id}: {label}"]
        if self.heading:
            parts.append(f'heading "{self.heading}"')
        if self.elements:
            parts.append("contains " + ", ".join(self.elements))
        return " — ".join(parts)


@dataclass(frozen=True, slots=True)
class RedesignInventory:
    sections: tuple[RedesignSection, ...] = ()
    #: Page-level features the redesign HTML demonstrably contains. ``None`` when the
    #: HTML could not be read — UNKNOWN, so no page-content claim is judged (a claim
    #: cannot be refuted by an inventory nobody took). Server-side claims still are.
    features: frozenset[str] | None = None

    @property
    def section_ids(self) -> frozenset[str]:
        return frozenset(s.id for s in self.sections)

    def prompt_block(self) -> str:
        if not self.sections:
            return (
                "REDESIGN CONTENT: no per-section inventory is available. "
                f"{_GENERAL_VISUAL_TERMS} — never claim a specific feature or fix.\n"
            )
        lines = "\n".join(f"  {i + 1}. {s.describe()}" for i, s in enumerate(self.sections))
        absent = ""
        if self.features is not None:
            missing = sorted(_FEATURE_LABELS[f] for f in _PAGE_FEATURES - self.features)
            if missing:
                absent = f"The redesign does NOT contain: {', '.join(missing)}.\n"
        if inventory_is_band_only(self):
            # Band-only inventory (prod r3b: band-01, band-02): no landmark names and no
            # heading/element read. "Describe ONLY what is listed" over a list of bare
            # positions plus a does-NOT-contain list leaves an empty script as the only
            # fully compliant answer. Named sections keep the strict rule below — their
            # names (hero, diensten, cta) are something the script can ground on.
            return (
                "REDESIGN CONTENT — the redesigned page is shown as these rendered "
                "screens, top to bottom (this is everything the viewer will see):\n"
                f"{lines}\n"
                f"{absent}"
                "GROUNDING RULES: no headings or elements could be read for these "
                f"screens. {_GENERAL_VISUAL_TERMS}, screen by screen — the after act "
                "is still required, an empty script is never correct. Never claim a "
                "specific feature, element or fix — a critique RECOMMENDATION is not "
                "something the redesign did. "
                f"{_SERVER_RULE}\n"
            )
        return (
            "REDESIGN CONTENT — the redesigned page consists of EXACTLY these rendered "
            "sections, top to bottom (this is everything the viewer will see):\n"
            f"{lines}\n"
            f"{absent}"
            "GROUNDING RULES: describe ONLY what is listed above. Never claim the "
            "redesign has a feature, element or fix that is not listed — a critique "
            f"RECOMMENDATION is not something the redesign did. {_SERVER_RULE}\n"
        )


def inventory_is_band_only(inventory: RedesignInventory) -> bool:
    """True when EVERY section is a content-less html_render viewport band.

    The ONE band-only predicate — :meth:`RedesignInventory.prompt_block` and the
    script stage's grounding re-ask both read it, so the first ask and the re-ask
    can never disagree about which page they are describing. A section counts only
    when its name is ``band-NN`` AND no heading/element was read: a named section
    with no readable HTML (``hero``, ``about-this-domain``) is NOT band-only — its
    name is something the script can ground on. No sections ⇒ ``False``.
    """
    return bool(inventory.sections) and all(
        s.band_number is not None and not s.has_content for s in inventory.sections
    )


_BAND_NAME_RE: Final = re.compile(r"band-(\d+)")
_GENERAL_VISUAL_TERMS: Final = (
    "Describe the redesign only in general visual terms (layout, hierarchy, typography, clarity)"
)
_SERVER_RULE: Final = (
    "The redesign is a static page design: it cannot fix server settings, so never "
    "say security headers, HTTPS/TLS or other server configuration are added, fixed "
    "or live."
)

_PAGE_FEATURES: Final = frozenset(
    {
        "form",
        "search",
        "chat",
        "cookie_banner",
        "video",
        "map",
        "breadcrumb",
        "skip_link",
        "booking",
    }
)
_FEATURE_LABELS: Final = {
    "form": "a form (contact/sign-up)",
    "search": "a search field",
    "chat": "a chat widget",
    "cookie_banner": "a cookie banner",
    "video": "an embedded video",
    "map": "an embedded map",
    "breadcrumb": "breadcrumb navigation",
    "skip_link": "a skip-to-content link",
    "booking": "a booking/scheduling widget",
    "security_headers": "security headers",
    "https": "HTTPS/TLS configuration",
}

_ORDER_RE = re.compile(r"/(\d+)_[^/]*$")


def _section_visuals(redesign: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], str, int]]:
    """``(visual, section_name, order)`` for each rendered section still, page order."""
    found: list[tuple[Mapping[str, Any], str, int]] = []
    for index, visual in enumerate(redesign.get("visuals") or []):
        if not isinstance(visual, dict):
            continue
        role = str(visual.get("role") or "")
        if not (
            visual.get("id") and visual.get("image_url") and role.startswith(SECTION_ROLE_PREFIX)
        ):
            continue
        match = _ORDER_RE.search(str(visual.get("image_url")))
        order = int(match.group(1)) if match else 1000 + index
        found.append((visual, role.removeprefix(SECTION_ROLE_PREFIX), order))
    return sorted(found, key=lambda item: item[2])


def sections_from_visuals(redesign: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """``(visual_id, section_name, order)`` for each rendered section still, page order.

    The visuals list is NOT in page order (prod: cta first, footer third); the still's
    file name (``07_cta.png``) is, so it decides when present.
    """
    return [(str(v["id"]), name, order) for v, name, order in _section_visuals(redesign)]


def rendered_section_visuals(redesign: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``(inventory_id, visual)`` per rendered section still, page order.

    ``inventory_id`` is the id :func:`build_inventory` gives the section (a repeated
    visual id renamed ``-2``, ``-3``) — the id the script grounds lines on. The
    ``visual`` keeps its own ``id``, which is what compile resolves a still by, so a
    consumer holding an inventory id can always reach the still it came from.
    Silent: the rename is warned about once, by :func:`build_inventory`.
    """
    found = _section_visuals(redesign)
    unique = _unique_ids([str(v["id"]) for v, _, _ in found])
    return [(uid, v) for uid, (v, _, _) in zip(unique, found, strict=True)]


class _RedesignHTMLParser(HTMLParser):
    """Headings, key elements and page features per ``data-section`` block."""

    _HEADINGS: Final = ("h1", "h2", "h3")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current: str | None = None
        self.headings: dict[str, str] = {}
        self.elements: dict[str, list[str]] = {}
        self.features: set[str] = set()
        self._heading_tag: str | None = None
        self._heading_text: list[str] = []
        self._button_text: list[str] | None = None

    def _add(self, element: str) -> None:
        if self.current is None:
            return
        bucket = self.elements.setdefault(self.current, [])
        if element not in bucket and len(bucket) < 6:
            bucket.append(element)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}
        blob = " ".join(attr.values()).lower()
        if attr.get("data-section"):
            self.current = attr["data-section"]
        if (
            tag in self._HEADINGS
            and self._heading_tag is None
            and self.current not in self.headings
        ):
            self._heading_tag, self._heading_text = tag, []
        self._element(tag, attr, blob)
        self._attribute_features(tag, attr, blob)

    def _element(self, tag: str, attr: dict[str, str], blob: str) -> None:
        input_type = attr.get("type", "text").lower()
        if tag == "form":
            self.features.add("form")
            self._add("form")
        elif tag in ("input", "textarea", "select"):
            if input_type == "search" or "search" in blob or "zoek" in blob:
                self.features.add("search")
                self._add("search field")
            elif tag == "textarea" or input_type in ("text", "email", "tel"):
                self.features.add("form")
                self._add("form fields")
            if input_type in ("date", "datetime-local"):
                self.features.add("booking")
        elif tag == "video" or (tag == "iframe" and re.search(r"youtube|vimeo", blob)):
            self.features.add("video")
            self._add("video")
        elif tag == "iframe" and "map" in blob:
            self.features.add("map")
            self._add("map")
        elif tag in ("button", "a") and (tag == "button" or re.search(r"\b(btn|button|cta)", blob)):
            self._button_text = []

    def _attribute_features(self, tag: str, attr: dict[str, str], blob: str) -> None:
        if attr.get("role") == "search":
            self.features.add("search")
        if "chat" in blob:
            self.features.add("chat")
        if "cookie" in blob:
            self.features.add("cookie_banner")
        if "breadcrumb" in blob:
            self.features.add("breadcrumb")
        if tag == "a" and "skip" in blob:
            self.features.add("skip_link")
        if re.search(r"calendly|booking|calendar", blob):
            self.features.add("booking")

    def handle_endtag(self, tag: str) -> None:
        if tag == self._heading_tag and self.current is not None:
            text = " ".join("".join(self._heading_text).split())
            if text:
                self.headings[self.current] = text[:80]
            self._heading_tag = None
        if tag in ("button", "a") and self._button_text is not None:
            text = " ".join("".join(self._button_text).split())
            if text:
                self._add(f'button "{text[:40]}"')
            self._button_text = None

    def handle_data(self, data: str) -> None:
        if self._heading_tag is not None:
            self._heading_text.append(data)
        if self._button_text is not None:
            self._button_text.append(data)


def _unique_ids(ids: list[str]) -> list[str]:
    """``ids`` with every repeat suffixed ``-2``, ``-3`` in order (pure, no logging).

    The first occurrence keeps its id; a suffix never reuses an id already present
    in the input.
    """
    taken = set(ids)
    seen: set[str] = set()
    out: list[str] = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
            continue
        n = 2
        while f"{vid}-{n}" in taken:
            n += 1
        new_id = f"{vid}-{n}"
        taken.add(new_id)
        seen.add(new_id)
        out.append(new_id)
    return out


def _dedupe_section_ids(rendered: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    """Make every visual id unique — repeats get ``-2``, ``-3`` in page order.

    ``RedesignInventory.section_ids`` is a set: a repeated id made the later section
    indistinguishable from the first, so no line could ever be grounded on it and its
    copy was silently lost. The renamed id is only the inventory's handle: consumers
    that need the still map it back with :func:`rendered_section_visuals`.
    """
    unique = _unique_ids([vid for vid, _, _ in rendered])
    out: list[tuple[str, str, int]] = []
    for new_id, (vid, name, order) in zip(unique, rendered, strict=True):
        if new_id != vid:
            log.warning(
                "website_review grounding: duplicate redesign section id %r (section %r) "
                "renamed to %r",
                vid,
                name,
                new_id,
            )
        out.append((new_id, name, order))
    return out


def build_inventory(redesign: Mapping[str, Any], html: str | None) -> RedesignInventory:
    """The rendered sections, enriched from ``html`` when it could be read."""
    rendered = _dedupe_section_ids(sections_from_visuals(redesign))
    if not html:
        return RedesignInventory(
            sections=tuple(RedesignSection(id=vid, name=name) for vid, name, _ in rendered),
            features=None,
        )
    # ``HTMLParser`` is lenient by design: malformed markup yields fewer facts, not
    # an exception.
    parser = _RedesignHTMLParser()
    parser.feed(html)
    parser.close()
    return RedesignInventory(
        sections=tuple(
            RedesignSection(
                id=vid,
                name=name,
                heading=parser.headings.get(name, ""),
                elements=tuple(parser.elements.get(name, ())),
            )
            for vid, name, _ in rendered
        ),
        features=frozenset(parser.features),
    )


# ---------------------------------------------------------------------------
# post-check
# ---------------------------------------------------------------------------

#: A feature CLAIM, by phrase. Deliberately narrow: "search" alone matches "search
#: engines" (an SEO finding), "headers" alone the page header section.
_CLAIMS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("form", re.compile(r"\bforms?\b", re.IGNORECASE)),
    (
        "search",
        re.compile(
            r"\bsearch\s+(?:bar|box|field|feature|function(?:ality)?|icon|input|button)\b"
            r"|\bsite[- ]search\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chat",
        re.compile(
            r"\blive[- ]?chat\b|\bchat\s*(?:widget|box|bot|window|button)\b|\bchatbot\b", re.I
        ),
    ),
    (
        "cookie_banner",
        re.compile(
            r"\bcookie\s+(?:banner|consent|notice|bar|pop[- ]?up)\b|\bconsent\s+banner\b", re.I
        ),
    ),
    ("video", re.compile(r"\b(?:embedded|hero|background|explainer|intro)\s+video\b", re.I)),
    ("map", re.compile(r"\b(?:embedded|interactive|google)\s+maps?\b", re.I)),
    ("breadcrumb", re.compile(r"\bbreadcrumbs?\b", re.I)),
    ("skip_link", re.compile(r"\bskip(?:[- ]to[- ](?:main[- ])?content|[- ]links?)\b", re.I)),
    (
        "booking",
        re.compile(
            r"\b(?:booking|scheduling|appointment)\s+(?:widget|calendar|system|tool)\b"
            r"|\bonline\s+booking\b|\bcalendly\b",
            re.I,
        ),
    ),
    # Server-side claims are PHRASES, never bare words (en + nl). The page under review
    # can legitimately be ABOUT certificates (cliffordtraining.nl sells certified
    # trainings: "certificates in executive coaching", "certificaten") and every page
    # has a header ("the header now shows the phone number", "section headers") —
    # neither is a claim about server configuration.
    (
        "security_headers",
        re.compile(
            r"\bsecurity[- ]?headers?\b|\bbeveiligings-?headers?\b"
            r"|\b(?:http|https|response|server)[- ]headers?\b"
            r"|\bcontent[- ]security[- ]policy\b|\bCSP\b|\bHSTS\b"
            r"|\bstrict[- ]transport[- ]security\b|\bx-frame-options\b"
            r"|\bx-content-type-options\b|\breferrer[- ]policy\b|\bpermissions[- ]policy\b",
            re.I,
        ),
    ),
    (
        "https",
        re.compile(
            # SSL/TLS also covers "SSL certificate", "TLS-certificaat", "SSL-certificaten".
            r"\bHTTPS\b|\bTLS\b|\bSSL\b"
            r"|\b(?:security|server|site|website|domain)[- ]certificates?\b"
            r"|\bbeveiligings-?certifica(?:at|ten)\b"
            r"|\b(?:secure|encrypted)\s+connections?\b"
            r"|\b(?:beveiligde|versleutelde)\s+verbinding(?:en)?\b",
            re.I,
        ),
    ),
)

#: A clause that DENIES or merely LACKS the feature is not a claim that it exists.
_NEGATION_RE: Final = re.compile(
    r"\b(?:no|not|never|without|missing|lacks?|lacking|none|absent|instead of|"
    r"cannot|can't|won't|isn't|aren't|doesn't|don't|still needs?|needs? to|should|"
    r"would|could|recommend\w*)\b|n't\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE: Final = re.compile(r"[.;:!?,\u2014\u2013]|\s-\s|\band\b|\bbut\b", re.IGNORECASE)

#: A BEFORE-act line narrating a fix — the before act plays over the ORIGINAL page,
#: which has none of them.
_FIX_RE: Final = re.compile(
    r"\b(?:we|i|we've|i've|we have|i have|the redesign|the new (?:design|layout|page|site|version))"
    r"\s+(?:added|fixed|moved|replaced|introduced|built|rebuilt|deferred|embedded|made|put|placed|"
    r"removed|swapped|solved|resolved|cleaned|tightened|adds|fixes|moves|puts|brings|embeds|replaces)\b"
    r"|\bnow\s+(?:it(?:'s|\s+is)|there(?:'s|\s+is)|it\s+sits|it\s+lives|it\s+loads|it\s+renders)\b"
    r"|\bit(?:'s|\s+is)\s+now\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScriptLineView:
    """The fields the post-check reads — decoupled from the stage's pydantic model."""

    text: str
    phase: str
    section_id: str | None = None


@dataclass(slots=True)
class ScriptViolations:
    #: ``(line index, feature)`` — a feature claimed for the redesign it does not contain.
    ungrounded_claims: list[tuple[int, str]] = field(default_factory=list)
    #: Indices of BEFORE-act lines narrating a fix over the original page.
    before_act_fixes: list[int] = field(default_factory=list)
    #: Act IMBALANCE, as sentences — the after act (the redesign) is too thin for
    #: the sections that were rendered, or has more lines than there are sections.
    imbalance: list[str] = field(default_factory=list)
    #: After-act lines not tied to a distinct rendered section, as sentences. A
    #: re-ask trigger ONLY: the visual director still maps such a line, so it is
    #: never recorded on the run.
    mapping: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.count() > 0

    def count(self) -> int:
        return (
            len(self.ungrounded_claims)
            + len(self.before_act_fixes)
            + len(self.imbalance)
            + len(self.mapping)
        )

    def grounding_problems(self, texts: Sequence[str]) -> list[str]:
        """The violations that make the video SAY something the picture does not show."""
        out = [
            f"line {i + 1} claims {_FEATURE_LABELS[feat]}, which the redesign does not "
            f'contain: "{texts[i]}"'
            for i, feat in self.ungrounded_claims
        ]
        out.extend(
            f'before-act line {i + 1} describes a fix over the ORIGINAL page: "{texts[i]}"'
            for i in self.before_act_fixes
        )
        return out

    def describe(self, texts: Sequence[str]) -> list[str]:
        return [*self.grounding_problems(texts), *self.imbalance, *self.mapping]


def _claims(text: str) -> list[str]:
    features: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text):
        if not clause.strip() or _NEGATION_RE.search(clause):
            continue
        for feature, pattern in _CLAIMS:
            if pattern.search(clause) and feature not in features:
                features.append(feature)
    return features


def find_script_violations(
    lines: Sequence[ScriptLineView],
    inventory: RedesignInventory,
    *,
    authored_before: int = 0,
    after_allowance: int | None = None,
) -> ScriptViolations:
    """Every way ``lines`` narrates something the redesign video will not show.

    ``authored_before`` is the length of an operator-authored before act (shotlist):
    those lines are the operator's own words and are not judged, and the act-balance
    rule does not apply (the before act's length is not the model's choice).

    ``after_allowance`` is the after-act line count the PROMPT stated when it is
    larger than the section count (``script.after_line_allowance`` — a page with few
    sections and an order that needs more narration than one line per section
    carries). The script is judged against the rule it was given: up to that many
    after-act lines, several of which may name the same section. ``None`` (or a
    value no larger than the section count) is the one-line-per-section rule.
    """
    found = ScriptViolations()
    features = inventory.features
    for index, line in enumerate(lines):
        if index < authored_before:
            continue
        if _is_after(line):
            found.ungrounded_claims.extend(
                (index, feature)
                for feature in _claims(line.text)
                if feature in SERVER_SIDE_FEATURES
                or (features is not None and feature not in features)
            )
        elif _FIX_RE.search(line.text):
            found.before_act_fixes.append(index)
    if inventory.sections:
        _judge_after_act(found, lines, inventory, authored_before, after_allowance)
    return found


def _is_after(line: ScriptLineView) -> bool:
    return (line.phase or "").strip().lower() == "after"


def _judge_after_act(
    found: ScriptViolations,
    lines: Sequence[ScriptLineView],
    inventory: RedesignInventory,
    authored_before: int,
    after_allowance: int | None = None,
) -> None:
    """The after act measured against the RENDERED sections it is shown over.

    Only judged when sections were rendered: without per-section stills the after
    act is one full-page still, and there is nothing to balance or map against.
    """
    after = [line for line in lines if _is_after(line)]
    if authored_before == 0 and len(after) < MIN_AFTER_SHARE * len(lines):
        found.imbalance.append(
            f"the after act (the redesign) has {len(after)} of {len(lines)} lines — it "
            f"must carry at least {MIN_AFTER_SHARE:.0%}"
        )
    valid = inventory.section_ids
    sections = len(inventory.sections)
    # The prompt let several lines share a section only when it stated an allowance
    # ABOVE the section count; otherwise the rule it gave was one line per section.
    shared = after_allowance is not None and after_allowance > sections
    if shared and after_allowance is not None:
        if len(after) > after_allowance:
            found.imbalance.append(
                f"the after act has {len(after)} lines for {sections} rendered "
                f"section(s) — at most {after_allowance} after-act lines for this order"
            )
    elif len(after) > sections:
        found.imbalance.append(
            f"the after act has {len(after)} lines for {sections} "
            "rendered sections — at most one line per section"
        )
    mapped = [line.section_id for line in after if line.section_id in valid]
    if len(mapped) < len(after):
        found.mapping.append(
            f"{len(after) - len(mapped)} after-act line(s) carry no rendered section_id"
        )
    if not shared and len(set(mapped)) < len(mapped):
        found.mapping.append("two after-act lines point at the same section")
