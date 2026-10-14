"""p0 of labs.infer: the reference model's inference path behind the Engine API, nothing optimized.

The evolved program is this whole file. The harness (protected) loads the checkpoint and calls only:

    engine = Engine(weights, config, device)   # weights: state dict of fp32 CPU tensors; config: dict
    logits = engine.prefill(ids)               # ids (1, T) int64 on device, 1 <= T <= block_size. Resets the
                                               #   state, returns (1, vocab) logits for the last position
    logits = engine.step(token)                # token (1, 1) int64 on device: append it, return (1, vocab)
    out = engine.generate(ids, n)              # (1, n) greedy tokens continuing ids (prefill included)

Positions past block_size roll: RoPE keys stay valid, so the cache is a ring of block_size slots and each new
token attends to every filled slot (as mini_llm's next_token_logits does at the model's own block_size).
Same ops in the same order as the reference (labs/infer/reference.py), so fp32 logits match it closely.
"""

import torch
import torch.nn.functional as F


class Engine:
    def __init__(self, weights: dict, config: dict, device: torch.device):
        self.device = device
        self.n_layer, self.n_head = config["n_layer"], config["n_head"]
        self.n_embd, self.block_size = config["n_embd"], config["block_size"]
        self.head_size = self.n_embd // self.n_head
        w = {k: v.to(device) for k, v in weights.items()}
        self.wte = w["token_embedding_table.weight"]
        self.layers = []
        for i in range(self.n_layer):
            p = f"blocks.{i}."
            self.layers.append({
                "ln1": (w[p + "ln1.weight"], w[p + "ln1.bias"]),
                "qkv": w[p + "sa.qkv.weight"],
                "proj": (w[p + "sa.proj.weight"], w[p + "sa.proj.bias"]),
                "ln2": (w[p + "ln2.weight"], w[p + "ln2.bias"]),
                "fc": (w[p + "ffwd.net.0.weight"], w[p + "ffwd.net.0.bias"]),
                "out": (w[p + "ffwd.net.2.weight"], w[p + "ffwd.net.2.bias"]),
            })
        self.ln_f = (w["ln_f.weight"], w["ln_f.bias"])
        self.lm_head = (w["lm_head.weight"], w["lm_head.bias"])
        # RoPE speeds and angle tables exactly as the reference builds them (float64 angles, stored float32)
        pairs = torch.arange(0, self.head_size // 2, 1, dtype=torch.float64)
        self.speeds = 10000 ** (-2 * pairs / self.head_size)
        self._table(self.block_size)
        shape = (1, self.n_head, self.block_size, self.head_size)
        self.k_cache = [torch.empty(shape, device=device) for _ in range(self.n_layer)]
        self.v_cache = [torch.empty(shape, device=device) for _ in range(self.n_layer)]
        self.pos = 0

    def _table(self, n: int) -> None:
        angles = torch.outer(torch.arange(n, dtype=torch.float64), self.speeds)
        self.cos, self.sin = torch.cos(angles).float().to(self.device), torch.sin(angles).float().to(self.device)

    @staticmethod
    def _rope(x, cos, sin):
        ret = torch.empty_like(x)
        ret[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        ret[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return ret

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        start = self.pos
        if start + T > self.cos.size(0):
            self._table(max(2 * self.cos.size(0), start + T))
        cos, sin = self.cos[start : start + T], self.sin[start : start + T]
        x = F.embedding(ids, self.wte)
        for i, L in enumerate(self.layers):
            h = F.layer_norm(x, (self.n_embd,), *L["ln1"])
            q, k, v = F.linear(h, L["qkv"]).view(B, T, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            s = start % self.block_size
            assert s + T <= self.block_size and (T == 1 or start < self.block_size), "prefill past the window"
            self.k_cache[i][:, :, s : s + T] = k
            self.v_cache[i][:, :, s : s + T] = v
            filled = min(start + T, self.block_size)
            k, v = self.k_cache[i][:, :, :filled], self.v_cache[i][:, :, :filled]
            out = F.scaled_dot_product_attention(q, k, v, is_causal=1 < T == filled)
            out = out.transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + F.linear(out, *L["proj"])
            h = F.layer_norm(x, (self.n_embd,), *L["ln2"])
            x = x + F.linear(F.gelu(F.linear(h, *L["fc"])), *L["out"])
        self.pos = start + T
        x = F.layer_norm(x[:, -1:], (self.n_embd,), *self.ln_f)
        return F.linear(x, *self.lm_head)[:, -1, :]

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        self.pos = 0
        return self._forward(ids)

    @torch.no_grad()
    def step(self, token: torch.Tensor) -> torch.Tensor:
        return self._forward(token)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        logits = self.prefill(ids)
        out = []
        for i in range(n):
            nxt = torch.argmax(F.softmax(logits, dim=-1), dim=-1, keepdim=True)  # (1, 1)
            out.append(nxt)
            if i + 1 < n:
                logits = self.step(nxt)
        return torch.cat(out, dim=1)
