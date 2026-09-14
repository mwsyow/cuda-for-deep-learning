import time

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PytorchTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(sequence_length, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                PytorchMHABlock(hidden_dim, num_heads, head_dim, sequence_length)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.fc_f = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list = None,
        use_cache: bool = False,
    ):
        # prefill -> x: (batch_size, sequence_length)
        # decode -> x: (batch_size, 1)
        _, sequence_length = x.shape
        past_length = 0
        if use_cache and kv_caches is not None:
            past_length = kv_caches[0][0].shape[1]

        if past_length + sequence_length > self.pos_emb.num_embeddings:
            raise ValueError(
                "The prompt and generated tokens exceed the configured sequence length"
            )

        pos_indices = torch.arange(
            past_length, past_length + sequence_length, device=x.device
        )

        tok = self.tok_emb(x)
        pos = self.pos_emb(pos_indices)
        x = tok + pos

        new_caches = []
        for i, block in enumerate(self.blocks):
            if kv_caches is not None:
                x, new_cache = block(x, kv_caches[i], use_cache)
            else:
                x, new_cache = block(x, use_cache=use_cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        x = self.fc_f(x)
        return x, new_caches

    def prefill(self, x):
        logits, new_cache = self.forward(x)
        return logits, new_cache

    def decode_step(self, x, kv_caches):
        logits, new_cache = self.forward(x, kv_caches, use_cache=True)
        return logits[:, -1:, :], new_cache


class PytorchMHABlock(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int, head_dim: int, sequence_length: int
    ):
        super().__init__()

        self.mha = PytorchMHA(hidden_dim, num_heads, head_dim, sequence_length)
        self.ln_mha = nn.LayerNorm(hidden_dim)
        self.ff = PytorchFeedForward(hidden_dim)
        self.ln_ff = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, kv_cache: dict = None, use_cache: bool = False):
        mha_out, new_cache = self.mha(x, kv_cache, use_cache)
        mha_out = self.ln_mha(mha_out)
        x = mha_out + x

        ff_out = self.ff(x)
        ff_out = self.ln_ff(ff_out)
        x = ff_out + x
        return x, new_cache


class PytorchMHA(nn.Module):
    def __init__(
        self, in_dim: int, num_heads: int, head_dim: int, sequence_length: int
    ):
        super().__init__()
        out_dim = num_heads * head_dim
        self.k = nn.Linear(in_dim, out_dim)
        self.q = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(-1)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.register_buffer(
            "tril", torch.tril(torch.ones(sequence_length, sequence_length))
        )
        self.fc = nn.Linear(out_dim, in_dim)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict = None,
        use_cache: bool = False,
    ):
        # prefill: x -> (B, sequence_length, hidden_dim)
        # decode: x -> (B, 1, hidden_dim)
        batch_size, sequence_length, hidden_dim = x.shape
        x_q = self.q(x)  # (B, sequence_length, hidden_dim) or (B, 1, hidden_dim)
        x_k = self.k(x)  # (B, sequence_length, out_dim) or (B, 1, hidden_dim)
        x_v = self.v(x)  # (B, sequence_length, out_dim) or (B, 1, hidden_dim)

        k_full = x_k
        v_full = x_v
        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_full = torch.cat((k_cache, x_k), dim=1)
            v_full = torch.cat((v_cache, x_v), dim=1)

        total_length = k_full.shape[1]
        k_head = k_full.reshape(
            batch_size, total_length, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (batch_size, num_heads, total_length, head_dim)
        v_head = v_full.reshape(
            batch_size, total_length, self.num_heads, self.head_dim
        ).transpose(1, 2)  # (batch_size, num_heads, total_length, head_dim)
        q_head = x_q.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # (batch_size, num_heads, sequence_length, head_dim) or (batch_size, num_heads, 1, head_dim)

        att_scores: torch.Tensor = (
            q_head @ k_head.transpose(-2, -1) * (self.head_dim**-0.5)
        )
        if not use_cache:
            att_scores = att_scores.masked_fill(
                self.tril[:sequence_length, :sequence_length] == 0, float("-inf")
            )
        att_weight = self.softmax(
            att_scores
        )  # (batch_size, num_heads, sequence_length, sequence_length) or (batch_size, num_heads, 1, sequence_length)
        out = (
            att_weight @ v_head
        )  # (batch_size, num_heads, sequence_length, head_dim) or (batch_size, num_heads, 1, head_dim)
        out = out.transpose(
            1, 2
        )  # (batch_size, sequence_length, num_heads, head_dim) or (batch_size, 1, num_heads, head_dim)
        out = out.contiguous().reshape(
            batch_size, sequence_length, self.head_dim * self.num_heads
        )  # (batch_size, sequence_length, out_dim) or (batch_size, 1, out_dim)
        out = self.fc(
            out
        )  # (batch_size, sequence_length, in_dim) or (batch_size, 1, in_dim)
        return out, (k_full, v_full)


class PytorchFeedForward(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


CHARS = "".join([chr(i) for i in range(32, 127)])
VOCAB_SIZE = len(CHARS)
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
ENCODE = lambda s: [STOI[c] for c in s if c in STOI]
DECODE = lambda l: "".join([ITOS[i] for i in l])


def generate_tokens(model, prompt_tokens, max_new_tokens, use_cache):
    generated = prompt_tokens.clone()

    with torch.inference_mode():
        if use_cache:
            logits, kv_caches = model.prefill(generated)
            for step in range(max_new_tokens):
                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1, keepdim=True
                )
                generated = torch.cat((generated, next_token), dim=1)
                if step + 1 < max_new_tokens:
                    logits, kv_caches = model.decode_step(next_token, kv_caches)
        else:
            for _ in range(max_new_tokens):
                logits, _ = model(generated, use_cache=False)
                next_token = torch.argmax(
                    logits[:, -1, :], dim=-1, keepdim=True
                )
                generated = torch.cat((generated, next_token), dim=1)

    return generated


def synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def benchmark_generation(model, prompt_tokens, max_new_tokens, use_cache):
    warmup_tokens = min(4, max_new_tokens)
    generate_tokens(model, prompt_tokens, warmup_tokens, use_cache)
    synchronize_device()

    start_time = time.perf_counter()
    generated = generate_tokens(model, prompt_tokens, max_new_tokens, use_cache)
    synchronize_device()
    elapsed = time.perf_counter() - start_time
    return generated, elapsed


if __name__ == "__main__":
    sequence_length = 256
    hidden_dim = 768
    num_layers = 16
    num_heads = 8
    head_dim = hidden_dim // num_heads
    seed = 42
    max_new_tokens = 200

    torch.manual_seed(seed)
    print("=== PyTorch Baseline ===")
    model = PytorchTransformer(
        VOCAB_SIZE, sequence_length, hidden_dim, num_layers, num_heads, head_dim
    )
    model = model.to(DEVICE)
    model.eval()

    prompt = "Once upon a time"

    prompt_tokens = torch.tensor([ENCODE(prompt)], dtype=torch.long, device=DEVICE)
    if prompt_tokens.shape[1] + max_new_tokens > sequence_length:
        raise ValueError("sequence_length is too small for this benchmark")

    cached_tokens, cached_time = benchmark_generation(
        model, prompt_tokens, max_new_tokens, use_cache=True
    )
    uncached_tokens, uncached_time = benchmark_generation(
        model, prompt_tokens, max_new_tokens, use_cache=False
    )

    outputs_match = torch.equal(cached_tokens, uncached_tokens)
    generated_text = DECODE(cached_tokens[0].cpu().tolist())

    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text}")
    print(f"Outputs match: {outputs_match}")
    print()
    print(
        f"With KV cache:    {cached_time:.3f} s "
        f"({max_new_tokens / cached_time:.2f} tokens/s)"
    )
    print(
        f"Without KV cache: {uncached_time:.3f} s "
        f"({max_new_tokens / uncached_time:.2f} tokens/s)"
    )
    print(f"Speedup:          {uncached_time / cached_time:.2f}x")
