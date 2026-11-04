"""A serving engine for any open model, built from mini_llm outward. See docs/ROADMAP.md.

The split: models (engine/models/) are architectures only, a stateless
forward(input_ids, positions, kv_cache) -> logits; everything else (the KV cache, positions, sampling,
stopping, batching, scheduling) belongs to the engine. Each step of the roadmap is checked against
golden outputs recorded from mini_llm's own generation before any of this existed (engine.golden).

PROGRESS is the last roadmap step you've finished. Tests for later steps are skipped in normal runs, so
the suite stays green while you work; run them anyway with --engine-todo:

    uv run pytest tests/test_engine_step1.py --engine-todo     # work on step 1
    # ...when it passes, set PROGRESS = 1 here: from then on step 1 runs in every test run.
"""

from engine.engine import Engine, RequestOutput
from engine.kv_cache import KVCache
from engine.sampling import Sampler, SamplingParams

PROGRESS = 0

__all__ = ["Engine", "KVCache", "PROGRESS", "RequestOutput", "Sampler", "SamplingParams"]
