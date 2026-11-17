"""Deps-aware stage factories shared by several blueprints, wrapped in
:class:`StageFactory` so every blueprint stage stays introspectable
(``stage_handler_class`` — the P8 requirements manifest reads each stage's
``provider_slots`` through it)."""

from __future__ import annotations

from src.services.blueprints.base import StageFactory
from src.services.stages.action_reel.footage_probe import (
    FootageProbeStage,
    make_footage_probe_stage,
)
from src.services.stages.generic.source_ingest import SourceIngestStage, make_source_ingest_stage

SOURCE_INGEST_FACTORY = StageFactory(SourceIngestStage, make_source_ingest_stage)
FOOTAGE_PROBE_FACTORY = StageFactory(FootageProbeStage, make_footage_probe_stage)
