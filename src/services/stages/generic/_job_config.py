"""Shared job-config whitelist builder for the AI-generation stages.

``ai_image_gen`` and ``ai_video_gen`` both translate their stage ``cfg`` into a
downstream ACG job_config by copying a whitelisted set of keys (dropping
``None`` values). The two stages share a common BASE set of parameters and each
adds its own stage-specific extras.

Defining the base **once** here is the whole point: a new common parameter
added to :data:`BASE_JOB_CONFIG_KEYS` is picked up by both stages at the same
time and can no longer be added to one stage's tuple while being silently
forgotten in the other (the old drift bug, where a forgotten common key was
dropped from the job_config with no error).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Parameters common to EVERY AI-generation stage. Add shared keys HERE, once,
# so image and video stay in lockstep and cannot drift.
BASE_JOB_CONFIG_KEYS: tuple[str, ...] = (
    "provider",
    "style",
    "negative_prompt",
    "extra_params",
)


def build_job_config(cfg: Mapping[str, Any], extra_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Whitelist ``cfg`` down to the shared base keys plus per-stage ``extra_keys``.

    A key whose value is ``None`` (or absent) is dropped, matching the prior
    per-stage behavior. The returned dict's key set is exactly the union of
    ``BASE_JOB_CONFIG_KEYS`` and ``extra_keys``, intersected with the
    non-``None`` keys present in ``cfg``.
    """
    keys = (*BASE_JOB_CONFIG_KEYS, *extra_keys)
    return {key: cfg[key] for key in keys if cfg.get(key) is not None}
