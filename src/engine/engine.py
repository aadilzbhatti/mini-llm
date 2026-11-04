"""Steps 2-3 (then 5): the generation loop, owned by the engine.

Step 2: one prompt at a time. Replaces mini_llm's model.generate / report.generate_until_eos: prefill
the prompt into a KVCache, then decode one token per step (sampler, stop at EOS or max_tokens), with
the rolling-window policy that mini_llm's next_token_logits implements today now living here.

Step 3: several prompts of different lengths in one batch (static batching): per-sequence positions
and lengths, sequences that finish early. Outputs must match each prompt run alone.

Step 5 turns this into a loop over a request queue (continuous batching); keep generate() as the
simple entry point on top of it.

The contract the tests rely on:

    eng = Engine(model, block_size=1024, tokenizer=tok, eos_token_id=50256)
    outs = eng.generate([prompt_ids_or_text, ...], SamplingParams(max_tokens=48, temperature=0.7, top_k=40, seed=0))
    outs[i].token_ids, outs[i].text, outs[i].finish_reason   # "eos" | "length"
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.sampling import SamplingParams


@dataclass
class RequestOutput:
    prompt_token_ids: list[int]
    token_ids: list[int]  # generated tokens only (EOS not included)
    finish_reason: str  # "eos" | "length"
    text: str | None = None  # decoded token_ids, when the engine has a tokenizer


class Engine:
    def __init__(self, model, block_size: int, tokenizer=None, eos_token_id: int | None = None, device="cpu"):
        raise NotImplementedError("engine step 2: the generation loop")

    def generate(
        self, prompts: list[str] | list[list[int]], params: SamplingParams | list[SamplingParams]
    ) -> list[RequestOutput]:
        raise NotImplementedError("engine step 2")
