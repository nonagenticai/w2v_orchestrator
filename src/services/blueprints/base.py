"""DAGBlueprint — declarative stage topology per project type."""

from __future__ import annotations

import enum
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from src.services.stages.base import StageHandler


class DeliverableProbe(enum.StrEnum):
    """How ``run_finalizer`` can ask "does this blueprint's deliverable exist?".

    The handle a terminal stage publishes is NOT always an object-store URL, and
    the two kinds need genuinely different probes:

    * ``OBJECT_STORE`` — the field holds a storage URL we minted (a presigned
      MinIO GET, or ``file://`` on the local backend). ``StorageService.object_exists``
      re-derives ``{bucket}/{key}`` and ``stat_object``s it with our own creds.
    * ``DESIGN_JOB`` — the field holds an ``html_render_service`` job id. The
      user-facing links for such a blueprint (``/api/redesign/jobs/{id}/html``,
      ``…/pdf``) are ORCHESTRATOR PROXY ROUTES, not storage URLs, so an
      object-store probe of them is permanently indeterminate — it can never
      answer, which is the same as no probe at all. "Does the deliverable exist"
      here means "does html_render still serve this job's HTML", so the probe
      asks html_render.

    A blueprint whose deliverable is neither cannot be verified from the
    orchestrator; it must add a member here rather than silently mis-declaring
    one of these two.
    """

    OBJECT_STORE = "object_store"
    DESIGN_JOB = "design_job"


@dataclass(frozen=True)
class StageFactory:
    """A ``handler_factory`` that also REMEMBERS which ``StageHandler`` it builds.

    A bare closure (the old shape) is opaque: nothing can ask a StageSpec what
    output model its stage declares without CALLING the factory, and calling one
    at wiring time is not safe — several handlers read settings in ``__init__``
    (``RedisSettings`` raises at import-time construction). Carrying the class
    alongside the callable makes the deliverable-wiring check
    (``DAGBlueprint.validate_deliverable_wiring``) a pure attribute read.

    Hand-written deps-aware factories wrap themselves in this instead of being
    passed bare, so they are introspectable too.
    """

    stage_cls: type[StageHandler]
    build: Callable[[dict], StageHandler]

    def __call__(self, deps: dict) -> StageHandler:
        return self.build(deps)


def handler(cls: type[StageHandler]) -> StageFactory:
    """Adapt a no-deps ``StageHandler`` subclass into a ``StageSpec`` handler_factory.

    Most stages need nothing from the run-time deps mapping, so their factory is
    just ``lambda deps: Cls()``. This names that intent and keeps blueprints a
    flat declarative list. Stages that DO consume deps (``course_compile_lessons``
    pulls run_engine/redis/session_factory) keep a hand-written factory — wrapped
    in ``StageFactory`` so it stays introspectable.
    """

    def _make(_deps: dict) -> StageHandler:
        return cls()

    return StageFactory(stage_cls=cls, build=_make)


def stage_handler_class(spec: StageSpec) -> type[StageHandler] | None:
    """The ``StageHandler`` subclass a spec builds, or ``None`` if untagged.

    ``None`` means the spec's factory is a bare callable (a test fixture, or a
    hand-written factory nobody wrapped) — the caller must treat that as
    "cannot introspect", never as "nothing declared".
    """
    factory = spec.handler_factory
    return factory.stage_cls if isinstance(factory, StageFactory) else None


@dataclass(frozen=True)
class StageSpec:
    name: str
    # Factory takes a deps mapping the dispatcher provides at run time. Stages
    # that need no deps ignore the argument; sub-DAG stages (course-level)
    # pull engine + dispatcher + redis + session_factory out of it.
    handler_factory: Callable[[dict], StageHandler]
    depends_on: tuple[str, ...] = ()
    parallel_group: str | None = None
    # Outer per-stage ceiling (TimeoutMiddleware), must cover the inner stage_call
    # worst case. ``None`` (the default) means "inherit the runtime base LLM ceiling"
    # — the DAG executor resolves it to PipelineSettings.llm_stage_base_ceiling_s
    # (env PIPELINE_LLM_STAGE_BASE_CEILING_S, default 700), so a slow-vLLM raise lands
    # in GitOps and lifts every base LLM stage at once. Stages that need a different
    # ceiling (redesign, record, child-run fanout, short deterministic stages) pin an
    # explicit int here. config._validate_llm_timeout_ladder keeps llm_budget_seconds
    # strictly below the resolved ceiling.
    default_timeout_seconds: int | None = None
    default_pause_after: bool = False

    # Allow lists in tests for ergonomics; coerce to tuple here.
    def __post_init__(self) -> None:
        if isinstance(self.depends_on, list):
            object.__setattr__(self, "depends_on", tuple(self.depends_on))


@dataclass
class DAGBlueprint:
    id: str
    project_type: str
    stages: list[StageSpec] = field(default_factory=list)
    # The stage whose primary output is THE deliverable artifact for this
    # blueprint (the compiled video). The run finalizer probes that stage's
    # ``artifact_url`` before marking the run ``completed`` so a run can't go
    # green pointing at a non-existent object. ``None`` ⇒ no run-level artifact
    # gate (the per-stage ArtifactUrl + compile-response checks still apply).
    terminal_artifact_stage: str | None = None
    # WHICH field of that stage's output holds the deliverable handle. Most
    # blueprints publish a storage URL under ``artifact_url`` (the default), but
    # a terminal stage is free to name its deliverable something else — and when
    # it does, a probe hard-coded to ``artifact_url`` finds nothing and disables
    # itself for the entire blueprint, silently. That is exactly what happened to
    # both web-design blueprints, whose deliverable handle is ``redesign_job_id``.
    # ``validate_deliverable_wiring`` (run by ``BlueprintRegistry.register``)
    # rejects a blueprint whose terminal stage does not DECLARE this field, so the
    # omission is a loud import-time failure instead of an inert runtime probe.
    terminal_artifact_field: str = "artifact_url"
    # HOW to check that handle exists. See ``DeliverableProbe``: an object-store
    # HEAD is the right question only for a handle that IS a storage URL.
    terminal_artifact_probe: DeliverableProbe = DeliverableProbe.OBJECT_STORE
    # The stage that RE-STITCHES the compiled MP4 — the cut point for surgical
    # timeline edits (reorder / renarrate / edit-clips re-run this stage onward;
    # see ``runs/elements.compile_cut_stage``). Normally this IS the
    # ``terminal_artifact_stage`` (the compile's output is the deliverable), so it
    # defaults to ``None`` ⇒ fall back to ``terminal_artifact_stage``. Set it
    # explicitly only when a post-compile stage (e.g. an accept-best footage
    # critique) is the deliverable but must NOT be the re-stitch anchor — the
    # re-stitch still has to happen at the real compile stage upstream of it.
    compile_cut_stage: str | None = None
    # The stage whose output holds the artifact the run actually DELIVERS.
    #
    # This is NOT ``terminal_artifact_stage``. That name is the RE-STITCH
    # ANCHOR / existence-probe target (``compile_cut_stage`` defaults to it, and
    # ``run_finalizer`` probes it), so on the video blueprints it deliberately
    # points at the COMPILE stage while the chain continues
    # compile → footage_critique → encode_master → final_critique. Every one of
    # those later stages may republish a DIFFERENT (re-mastered, accept-best)
    # artifact, so "the file the customer gets" is the MOST DOWNSTREAM
    # artifact-bearing stage, not the compile.
    #
    # Nothing could answer that question before: ``RunDetail`` carried no
    # delivered-artifact stage or URL, so an assertion about "the deliverable"
    # had to name a stage by hand — and naming an INTERMEDIATE one passes green
    # while the shipped master is broken.
    #
    # ``None`` ⇒ fall back to ``terminal_artifact_stage`` (see
    # ``delivered_stage``), so a blueprint whose compile IS its deliverable
    # changes in no way. Set it explicitly only where a pass-through stage
    # follows the cut. ⛔ Never "fix" this by moving ``terminal_artifact_stage``
    # instead — that would move the re-stitch cut and break reorder/renarrate.
    delivered_artifact_stage: str | None = None

    def __post_init__(self) -> None:
        names = {s.name for s in self.stages}
        if len(names) != len(self.stages):
            raise ValueError("duplicate stage names in blueprint")
        for s in self.stages:
            for dep in s.depends_on:
                if dep not in names:
                    raise ValueError(f"stage '{s.name}' lists unknown dependency '{dep}'")
        if self.terminal_artifact_stage is not None and self.terminal_artifact_stage not in names:
            raise ValueError(
                f"terminal_artifact_stage '{self.terminal_artifact_stage}' is not a stage"
            )
        if self.compile_cut_stage is not None and self.compile_cut_stage not in names:
            raise ValueError(f"compile_cut_stage '{self.compile_cut_stage}' is not a stage")
        if self.delivered_artifact_stage is not None and self.delivered_artifact_stage not in names:
            raise ValueError(
                f"delivered_artifact_stage '{self.delivered_artifact_stage}' is not a stage"
            )
        # NB: the deliverable-FIELD check is deliberately NOT here. It needs the
        # terminal stage's output model, which only exists for a real (tagged)
        # handler; ad-hoc blueprints built in tests from stub specs must stay
        # constructible. Registration is the real wiring moment — see
        # ``validate_deliverable_wiring`` and ``BlueprintRegistry.register``.
        # cycle detection
        order = self._topological_order_or_raise()
        self._cached_order = order

    def _topological_order_or_raise(self) -> list[str]:
        in_degree: dict[str, int] = {s.name: 0 for s in self.stages}
        adj: dict[str, list[str]] = defaultdict(list)
        for s in self.stages:
            for dep in s.depends_on:
                adj[dep].append(s.name)
                in_degree[s.name] += 1
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order: list[str] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)
        if len(order) != len(self.stages):
            raise ValueError("cycle in blueprint")
        return order

    def validate_deliverable_wiring(self) -> None:
        """Fail LOUDLY if this blueprint's deliverable could never be verified.

        Called by ``BlueprintRegistry.register``, i.e. at import of the blueprint
        module — so a new blueprint that names a ``terminal_artifact_stage`` whose
        output model does not declare ``terminal_artifact_field`` cannot reach
        production: the service refuses to start and CI goes red on the spot.

        This is the ROOT CAUSE fix. The pre-existing failure mode was not "the
        probe returned the wrong answer", it was "the probe was never armed and
        nothing said so": ``__post_init__`` checked only that the stage NAME
        exists, so a terminal stage publishing its deliverable under any other
        field name left the run-level existence gate permanently inert while every
        run still reported ``completed``.

        Raises ``ValueError`` for a stage whose handler class cannot be
        introspected too — an untagged factory is an unanswered question, and an
        unanswered question here silently degrades to "unverified". Wrap the
        factory in ``StageFactory`` rather than leaving the gap open.
        """
        # FIRST, and outside the terminal-stage early return: a blueprint may name
        # a delivered stage without a terminal one, and a typo there must still be
        # an import-time failure rather than a null served forever.
        self._validate_delivered_artifact_stage()
        if self.terminal_artifact_stage is None:
            return
        if self.terminal_artifact_probe not in tuple(DeliverableProbe):
            raise ValueError(
                f"blueprint '{self.id}' declares unknown terminal_artifact_probe "
                f"'{self.terminal_artifact_probe}'; known kinds: "
                f"{[p.value for p in DeliverableProbe]}"
            )
        spec = self.stage(self.terminal_artifact_stage)
        cls = stage_handler_class(spec)
        if cls is None:
            raise ValueError(
                f"blueprint '{self.id}': terminal stage "
                f"'{self.terminal_artifact_stage}' has an untagged handler_factory, "
                f"so its deliverable field cannot be verified at wiring time. Wrap "
                f"the factory: handler_factory=StageFactory(TheStage, _the_factory)."
            )
        if self.terminal_artifact_field not in cls.output_model.model_fields:
            raise ValueError(
                f"blueprint '{self.id}': terminal stage "
                f"'{self.terminal_artifact_stage}' output model "
                f"{cls.output_model.__name__} declares no "
                f"'{self.terminal_artifact_field}' field, so the run-level "
                f"deliverable-existence probe would never fire and every run of "
                f"this blueprint would report completed unverified. Either publish "
                f"that field from the stage, or set terminal_artifact_field (and "
                f"terminal_artifact_probe) to the handle the stage actually emits."
            )

    def _validate_delivered_artifact_stage(self) -> None:
        """Same loudness for ``delivered_artifact_stage`` as for the terminal one.

        ``__post_init__`` only proves the NAME exists. A stage that exists but
        publishes no ``terminal_artifact_field`` is worse than a typo: the API
        would serve ``delivered_artifact_url: null`` forever and every consumer
        would read that as "this run shipped nothing" rather than "this wiring is
        wrong". Both failures belong at import time, where CI sees them.

        Only checked when the blueprint SETS the field — an unset one resolves to
        ``terminal_artifact_stage``, which the caller already validated.
        """
        if self.delivered_artifact_stage is None:
            return
        spec = self.stage(self.delivered_artifact_stage)
        cls = stage_handler_class(spec)
        if cls is None:
            raise ValueError(
                f"blueprint '{self.id}': delivered stage "
                f"'{self.delivered_artifact_stage}' has an untagged handler_factory, "
                f"so the delivered artifact cannot be resolved at wiring time. Wrap "
                f"the factory: handler_factory=StageFactory(TheStage, _the_factory)."
            )
        if self.terminal_artifact_field not in cls.output_model.model_fields:
            raise ValueError(
                f"blueprint '{self.id}': delivered stage "
                f"'{self.delivered_artifact_stage}' output model "
                f"{cls.output_model.__name__} declares no "
                f"'{self.terminal_artifact_field}' field, so the run's delivered "
                f"artifact could never be resolved and every consumer would read "
                f"the run as having shipped nothing. Name a stage that publishes "
                f"that field, or leave delivered_artifact_stage unset to inherit "
                f"terminal_artifact_stage."
            )

    def delivered_stage(self) -> str | None:
        """The stage whose output holds the DELIVERED artifact.

        ``delivered_artifact_stage`` when set, else ``terminal_artifact_stage``
        — the same "explicit, else inherit" shape ``compile_cut_stage`` uses.
        ``None`` only when the blueprint declares no artifact at all.
        """
        return self.delivered_artifact_stage or self.terminal_artifact_stage

    def delivered_artifact_chain(self) -> tuple[str, ...]:
        """Ordered candidate stages for the delivered artifact — most downstream first.

        DERIVED from this blueprint's own DAG, never hand-listed: it is the
        delivered stage plus its transitive ANCESTORS, kept only where the
        stage's output model declares ``terminal_artifact_field``, in reverse
        topological order.

        That reproduces exactly the fallback chain the pipeline itself walks —
        ``final_critique._starting_artifact`` reads
        ``demo_video_encode_master`` → ``demo_video_footage_critique`` →
        ``demo_video_compile``, which is what this returns below
        ``final_critique`` — because each of those stages is skip-clean: it emits
        a falsy ``{}`` when it did not run, and the next one upstream owns the
        artifact. Deriving it rather than restating it is what stops the API and
        the pipeline disagreeing about which file shipped.

        Empty when the blueprint declares no artifact stage.
        """
        delivered = self.delivered_stage()
        if delivered is None:
            return ()
        eligible = {delivered} | self._ancestors(delivered)
        chain: list[str] = []
        for name in reversed(self.topological_order()):
            if name not in eligible:
                continue
            cls = stage_handler_class(self.stage(name))
            if cls is not None and self.terminal_artifact_field in cls.output_model.model_fields:
                chain.append(name)
        return tuple(chain)

    def _ancestors(self, name: str) -> set[str]:
        """All stages ``name`` transitively depends on, excluding ``name``."""
        by_name = {s.name: s for s in self.stages}
        if name not in by_name:
            raise KeyError(name)
        seen: set[str] = set()
        queue = deque(by_name[name].depends_on)
        while queue:
            n = queue.popleft()
            if n in seen:
                continue
            seen.add(n)
            queue.extend(by_name[n].depends_on)
        return seen - {name}

    def topological_order(self) -> list[str]:
        return list(self._cached_order)

    def parallel_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for s in self.stages:
            if s.parallel_group:
                groups[s.parallel_group].append(s.name)
        return dict(groups)

    def stage(self, name: str) -> StageSpec:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def descendants(self, name: str) -> set[str]:
        """All stages transitively reachable FROM ``name`` via depends_on edges
        (i.e. stages that (in)directly depend on ``name``), excluding ``name`` itself.
        Raises KeyError if ``name`` is not a stage."""
        names = {s.name for s in self.stages}
        if name not in names:
            raise KeyError(name)
        # Same forward adjacency (dep -> dependents) used by
        # _topological_order_or_raise, built locally so no cached state is mutated.
        adj: dict[str, list[str]] = defaultdict(list)
        for s in self.stages:
            for dep in s.depends_on:
                adj[dep].append(s.name)
        reached: set[str] = set()
        queue = deque([name])
        while queue:
            n = queue.popleft()
            for m in adj[n]:
                if m not in reached:
                    reached.add(m)
                    queue.append(m)
        return reached - {name}
