"""p0 of labs.decode: the owner's sampling protocol (temperature 0.7, top-k 40, multinomial), nothing else.

The evolved program is this whole file. For every sample the harness (protected) makes a fresh Sampler and calls
next_token once per new token; it owns the model, the KV cache and stopping at EOS:

    s = Sampler(vocab_size, eos_id, prompt_ids)          # prompt_ids: list[int]
    tok = s.next_token(logits, generated, generator)      # logits: (vocab,) float32 CPU tensor, the model's raw
                                                          #   next-token logits; generated: list[int] so far;
                                                          #   generator: a seeded CPU torch.Generator (use it for
                                                          #   all randomness); returns an int token id
"""

import torch

TEMPERATURE, TOP_K = 0.7, 40


class Sampler:
    def __init__(self, vocab_size: int, eos_id: int, prompt_ids: list[int]):
        self.vocab_size, self.eos_id = vocab_size, eos_id

    def next_token(self, logits: torch.Tensor, generated: list[int], generator: torch.Generator) -> int:
        logits = logits / TEMPERATURE
        kth = torch.topk(logits, TOP_K).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))
