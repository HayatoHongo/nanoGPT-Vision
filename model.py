# added top-p and top-k filtering in generate function
# set vocab_size in config.py
# MHA with KV cache + RoPE + PyTorch SDPA.
# This traditional implementation is easier to understand, and still efficient in practice.
# GQA and MLA is a great way for long-text inference with reduced KV cache size,
# but both comes with slight loss increase and no efficiency merits during training phase.
# KV cache does not help training speed. Codebase will be simpler without it.
# KV cache supports multi-turn continuation by RoPE with position offset.
# No Dropout. Dataset is large enough and regularization is not necessary.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim)
        # keep embedding in default dtype (autocast will handle bf16 when enabled)

    def forward(self, input_indices):
        return self.token_embedding_table(input_indices)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, rope_theta=1e6):
        super().__init__()

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2) / dim))
        position_index = torch.arange(max_seq_len)
        frequency_matrix = torch.einsum('i,j->ij', position_index, inv_freq)

        cosine = torch.cos(frequency_matrix)[None, None, :, :]
        sine = torch.sin(frequency_matrix)[None, None, :, :]

        self.register_buffer("cos_cached", cosine, persistent=False)
        self.register_buffer("sin_cached", sine, persistent=False)

    def apply_rotary_emb(self, x, position_offset=0):
        sequence_length = x.size(2)

        cosine = self.cos_cached[:, :, position_offset:position_offset + sequence_length, :]
        sine = self.sin_cached[:, :, position_offset:position_offset + sequence_length, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cosine - x_odd * sine
        rotated_odd = x_odd * cosine + x_even * sine

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = rotated_even
        rotated[..., 1::2] = rotated_odd

        return rotated

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.embed_dim = config.embedding_dim
        self.head_dim = self.embed_dim // self.num_heads

        # QKV projection
        self.query_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_fc   = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_fc = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Rotary Positional Embedding (RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_sequence_length,
            rope_theta=config.rope_theta
        )

        self.output_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(
                config.max_sequence_length,
                config.max_sequence_length,
                dtype=torch.bool
            )),
            persistent=False
        )

        # KV cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_pos = 0

    # --------------------------------------------------
    # router
    # --------------------------------------------------
    def forward(self, x, use_cache=False):
        input_len = x.size(1)
        if use_cache is False:
            return self.forward_no_cache(x)
        elif use_cache is True and input_len > 1:
            return self.forward_prefill(x)
        elif use_cache is True and input_len == 1: # Hi scenario also starts with T==1
            return self.forward_cached_decoding(x)
        else:
            raise RuntimeError("Unexpected condition in MultiHeadAttention forward")

    # --------------------------------------------------
    # (1) no cache : training 
    # --------------------------------------------------
    def forward_no_cache(self, x):
        B, T, C = x.shape

        Q = self.query_fc(x)
        K = self.key_fc(x)
        V = self.value_fc(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE : offset = 0
        Q = self.rotary_emb.apply_rotary_emb(Q, position_offset=0)
        K = self.rotary_emb.apply_rotary_emb(K, position_offset=0)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output_projection(out)
        return out

    # --------------------------------------------------
    # (2) prefill : initialize KV cache
    # --------------------------------------------------
    def forward_prefill(self, x):
        B, T, C = x.shape

        Q = self.query_fc(x)
        K = self.key_fc(x)
        V = self.value_fc(x)

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # init cache
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                B, self.num_heads, self.config.max_sequence_length, self.head_dim,
                device=x.device, dtype=K.dtype
            )
            self.cache_v = torch.zeros(
                B, self.num_heads, self.config.max_sequence_length, self.head_dim,
                device=x.device, dtype=V.dtype
            )
            self.current_pos = 0

        # RoPE : offset = current_pos (supports multi-turn continuation)
        Q = self.rotary_emb.apply_rotary_emb(Q, position_offset=self.current_pos)
        K = self.rotary_emb.apply_rotary_emb(K, position_offset=self.current_pos)

        # prevent overflow
        if self.current_pos + T > self.config.max_sequence_length:
            raise RuntimeError("KV cache exceeded max_sequence_length")

        self.cache_k[:, :, self.current_pos:self.current_pos + T, :] = K
        self.cache_v[:, :, self.current_pos:self.current_pos + T, :] = V

        K = self.cache_k[:, :, :self.current_pos + T, :]
        V = self.cache_v[:, :, :self.current_pos + T, :]

        attn_mask = self.causal_mask[
            self.current_pos : self.current_pos + T,
            : self.current_pos + T
        ]

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=False
        )

        self.current_pos += T

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output_projection(out)
        return out

    # --------------------------------------------------
    # (3) decode : cached decoding (1 token)
    # --------------------------------------------------
    def forward_cached_decoding(self, x):
        B, T, C = x.shape
        assert T == 1, "cached decoding expects T==1"

        Q = self.query_fc(x)
        K = self.key_fc(x)
        V = self.value_fc(x)

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # This is not usually needed since prefill should have initialized the cache.
        # Just in case for "Hi" scenario, which starts with single token input.
        if self.cache_k is None:
            self.cache_k = torch.zeros(
                B, self.num_heads, self.config.max_sequence_length, self.head_dim,
                device=x.device, dtype=K.dtype
            )
            self.cache_v = torch.zeros(
                B, self.num_heads, self.config.max_sequence_length, self.head_dim,
                device=x.device, dtype=V.dtype
            )
            self.current_pos = 0

        if self.current_pos + 1 >= self.config.max_sequence_length:
            raise RuntimeError("KV cache exceeded max_sequence_length")

        # RoPE : offset = current_pos
        Q = self.rotary_emb.apply_rotary_emb(Q, position_offset=self.current_pos)
        K = self.rotary_emb.apply_rotary_emb(K, position_offset=self.current_pos)

        self.cache_k[:, :, self.current_pos:self.current_pos + 1, :] = K
        self.cache_v[:, :, self.current_pos:self.current_pos + 1, :] = V

        K = self.cache_k[:, :, :self.current_pos + 1, :]
        V = self.cache_v[:, :, :self.current_pos + 1, :]

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=False
        )

        self.current_pos += 1

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output_projection(out)
        return out

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None
        self.current_pos = 0



class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()    
        self.net = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim, bias=False),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.multihead_attention = MultiHeadAttention(config=config)
        self.feed_forward = FeedForward(config=config)


    def forward(self, input_tensor, use_cache=False):
        normed_input = self.layer_norm1(input_tensor)
        attention_output = self.multihead_attention(normed_input, use_cache=use_cache)
        residual_attention = attention_output + input_tensor
        normed_attention = self.layer_norm2(residual_attention)
        feedforward_output = self.feed_forward(normed_attention)
        final_output = feedforward_output + residual_attention
        return final_output


class VocabularyLogits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.vocab_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, transformer_block_output):
        x = transformer_block_output
        normalized_output = self.output_norm(x)
        vocab_logits = self.vocab_projection(normalized_output)
        return vocab_logits


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_layer = TokenEmbedding(config=config)
        self.blocks = nn.ModuleList([TransformerBlock(config=config) for _ in range(config.layer_count)])
        self.vocab_projection = VocabularyLogits(config=config)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_indices, target_indices, use_cache=False):
        token_embeddings = self.token_embedding_layer.forward(input_indices)

        x = token_embeddings
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        logits = self.vocab_projection(x)

        if target_indices is None:
            return logits, None

        batch_size, token_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * token_len, vocab_size)
        targets_flat = target_indices.view(batch_size * token_len)
        loss = self.criterion(logits_flat, targets_flat)
        return logits, loss


    def generate(self,
        input_indices,
        max_new_tokens,
        temperature=1.0,
        use_cache=True,
        reset_cache=False,
        top_k=None,      # ### NEW ###
        top_p=None,      # ### NEW ###
    ):
        self.eval()

        if reset_cache:
            for block in self.blocks:
                block.multihead_attention.reset_cache()

        next_token = None

        for i in range(max_new_tokens):
            if use_cache:
                if i == 0:
                    logits, _ = self.forward(input_indices, None, use_cache=True)
                else:
                    logits, _ = self.forward(next_token, None, use_cache=True)
            else:
                logits, _ = self.forward(input_indices, None, use_cache=False)

            """ DELETE
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            """

            ### NEW ###
            last_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, last_logits.size(-1))
                values, _ = torch.topk(last_logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                last_logits = torch.where(
                    last_logits < min_value,
                    torch.full_like(last_logits, float("-inf")),
                    last_logits,
                )

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                sorted_logits = torch.where(
                    sorted_mask,
                    torch.full_like(sorted_logits, float("-inf")),
                    sorted_logits,
                )

                last_logits = torch.zeros_like(last_logits).scatter(
                    -1, sorted_indices, sorted_logits
                )

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ### NEW ###

            yield int(next_token.item())
            input_indices = torch.cat((input_indices, next_token), dim=1)