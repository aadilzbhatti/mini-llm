"""Step 2: sampling, owned by the engine instead of the model.

Today sampling lives in mini_llm (model._generate and report.generate_until_eos). The golden sampled
cases were made like this, per prompt (engine/golden.py has the exact text):

    torch.manual_seed(seed)               # once, before the first token
    each step: logits / temperature -> keep the top_k largest (rest -inf) -> softmax -> multinomial(1)
    greedy (temperature 0) = argmax; stop before appending EOS

To reproduce them token for token your sampler has to consume random numbers in exactly that order.
Later (batching, step 3) each request needs its own generator rather than the global seed: worth
thinking about now, since a batch of requests with different seeds can't share torch.manual_seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Per-request decoding settings (vLLM calls its equivalent the same)."""

    max_tokens: int = 16
    temperature: float = 0.0  # 0 = greedy
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop_at_eos: bool = True


class Sampler:
    """logits for the next token of each sequence -> the chosen token ids."""

    def __call__(
        self, logits: torch.Tensor, params: list[SamplingParams], generators: list[torch.Generator | None]
    ) -> torch.Tensor:
        raise NotImplementedError("engine step 2: the sampler")
