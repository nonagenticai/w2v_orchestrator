"""Generic narrated-video sub-chain shared by every scorecard audit type.

``seo_audit`` and ``cro_audit`` produce the SAME optional video from their scan
output (``script → slides → {narrate ‖ record} → compile``); only the script
prompt (breakdown key + a few labels) and the deck-title fallback genuinely
differ. That variance is class-level config on the base stages here — the
per-type stage modules under ``stages/{seo,cro}_audit/`` are thin declaration
subclasses that set ``name`` + those config attrs. Adding the video variant to a
new audit type is five one-line subclasses, not five cloned files.
"""
