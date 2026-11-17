"""Every concept-image submit/wait in ``src/`` goes through ONE helper.

``ai_image_gen``, ``course_thumbnail`` and the storyboard panel renderer each held
their own submit + wait, and each carried a render-sized deadline (180 s / 180 s
/ 300 s) that the one-at-a-time image backend's QUEUE blew through. The shared
helper ``src.services.concept_images.generate_concept_image`` owns the deadline
(caller's budget minus margin), the same-key retry and the final-attempt cancel.
This guard fails if any other module calls the ACG client's concept-image
job methods directly, or hand-builds a ``/api/concept-images`` request, so no
caller can quietly hold its own short deadline again.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
#: The client defines the methods; the helper is the one caller allowed.
_ALLOWED = {
    _SRC / "clients" / "ai_content_generation_client.py",
    _SRC / "services" / "concept_images.py",
}
_GUARDED_METHODS = frozenset(
    {
        "create_concept_image",
        "wait_for_concept_image",
        "get_concept_image_job",
        "cancel_concept_image",
    }
)
_GUARDED_PATH = "/api/concept-images"


def _violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _GUARDED_METHODS:
            found.append(f"{filename}:{node.lineno}: direct .{func.attr}(...)")
        # A raw ``submit_job("/api/concept-images", ...)`` / ``poll_job(...)`` /
        # ``_request("GET", f"/api/concept-images/...")`` bypasses the helper too.
        for arg in [*node.args, *(k.value for k in node.keywords)]:
            for sub in ast.walk(arg):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and sub.value.startswith(_GUARDED_PATH)
                ):
                    found.append(f"{filename}:{node.lineno}: raw {_GUARDED_PATH} request")
    return found


def test_only_the_shared_helper_calls_the_concept_image_job_methods() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path in _ALLOWED:
            continue
        offenders += _violations(path.read_text(encoding="utf-8"), str(path.relative_to(_SRC)))
    assert offenders == [], (
        "concept-image jobs must go through src.services.concept_images."
        "generate_concept_image (budget-sized deadline, same-key retry, final "
        "cancel):\n" + "\n".join(offenders)
    )


def test_the_guard_is_not_vacuous() -> None:
    # The helper itself is exactly the shape the guard must catch elsewhere.
    helper = (_SRC / "services" / "concept_images.py").read_text(encoding="utf-8")
    assert len(_violations(helper, "concept_images.py")) >= 3
    bypass = (
        "async def f(c):\n"
        "    j = await c.create_concept_image('p', {}, idempotency_key='k')\n"
        "    await c.wait_for_concept_image(j, timeout=180)\n"
        "    await c.poll_job('/api/concept-images/jobs', j, timeout=180)\n"
    )
    assert len(_violations(bypass, "bypass.py")) == 3
