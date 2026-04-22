import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
from collections import OrderedDict


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


def FeedForward(dmodel):
    original_hidden_dim = 4 * dmodel
    hidden_dim = int(original_hidden_dim * (2 / 3))

    class SwiGLU(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            return F.silu(x1) * x2

    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                ("pre_swiglu", nn.Linear(dmodel, 2 * hidden_dim, bias=True)),
                ("swiglu", SwiGLU()),
                ("post_swiglu", nn.Linear(hidden_dim, dmodel, bias=True)),
            ]
        )
    )


class IndigoAttentionLayer(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(IndigoAttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

        d_head = dmodel // heads
        self.relative_positional_embedding = nn.Embedding(3, d_head)
        # learnable embeddings for relative position from {-1, 0, 1}

    def forward(self, x, r, attention_mask):

        # r contains only {-1, 0, 1} values 
        # r[i, j] = -1 -> i'th token is left from j'th token
        # r[i, j] = 0 -> i == j
        # r[i, j] = 1 -> i'th token is right from j'th token

        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        R = self.relative_positional_embedding(r + 1)  # (batch, seq_len, seq_len, d_head), no in-place mutation

        q_dot_k = torch.matmul(query, key.transpose(-2, -1))
        q_dot_r = torch.einsum('bhid,bijd->bhij', query, R)
        S = q_dot_k + q_dot_r
        # instead of S[i, j] = Q[i] * K[j] in ordinary attention, we introduce relative position bias
        # S[i, j] = Q[i] * (K[j] + R[i, j])
        assert S.shape == (batch, self.heads, seq_len, seq_len)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len), True = padding (ignore)
            S = S.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = torch.softmax(((1 / (self.dmodel ** 0.5)) * S), dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        # I don't yet know, how to use attention backend for this

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output
    
class IndigoEncodingBlock(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = IndigoAttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, R, attention_mask):
        out_attention = self.attention_layer(x, R, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x, R

class IndigoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [IndigoEncodingBlock(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )


    def forward(self, input_ids, R, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output, R = block(output, R, attention_mask)

        return output, R

class IndigoWordDecodingHead(nn.Module):

    def __init__(self, dmodel, vocab_size, seq_len):
        super(IndigoWordDecodingHead, self).__init__()
        
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.proj = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, H, embedding_matrix, R):
        
        assert H.shape[:-2] == (self.seq_len, self.dmodel)
        assert embedding_matrix.shape[:-2] == (self.vocab_size, self.dmodel)
        assert R.shape[:-2] == (self.seq_len, self.seq_len) # relative position matrix

        output = torch.softmax(torch.matmul(self.proj(H[..., -1, :]), embedding_matrix.T), dim=-1)

        return output, R


class IndigoPositionDecodingHead(nn.Module):

    def __init__(self, dmodel, vocab_size, seq_len):
        super(IndigoPositionDecodingHead, self).__init__()
        
        self.dmodel = dmodel
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.state_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.left_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.right_proj = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, H, embedding_matrix, R, z):
        
        assert H.shape[:-2] == (self.seq_len, self.dmodel)
        assert embedding_matrix.shape[:-2] == (self.vocab_size, self.dmodel)
        assert R.shape[:-2] == (self.seq_len, self.seq_len) # relative position matrix

        left_positions = self.left_proj(H)
        right_positions = self.right_proj(H)

        query = (self.state_proj(H[..., -1, :]) + embedding_matrix[z]).unsqueeze(-2)
        keys = torch.cat([left_positions, right_positions], dim=-2).transpose(-1, -2)
        output = torch.softmax(torch.matmul(query, keys).squeeze(-2), dim=-1)

        return output, R 

def insert_relative_position_to_matrix(p, R):
    
    # extends relative positon matrix R with a new position p
    # p is an integer from range [0, 2n-1] 
    # R is an (n x n) matrix

    n = R.shape[-1] # length of the previously generated sequence
    neighbour_token = p % n 

    new_col = R[..., neighbour_token].clone()
    new_col[..., neighbour_token] = 1 if p < n else -1

    R_expanded = torch.cat([R, new_col.unsqueeze(-1)], dim=-1)

    new_row = -new_col.transpose(-1, -2)
    zero_pad = torch.zeros_like(new_row[..., :1])
    new_row = torch.cat([new_row, zero_pad], dim=-1)
    
    return torch.cat([R_expanded, new_row], dim=-2)


def restore_permutation(x, R):

    # permutes the sequence according to the relative position matrix

    R_clamped = torch.clamp(R, min=0)
    positions = torch.sum(R_clamped, dim=-1).long()
    sorted_indices = torch.argsort(positions, dim=-1)
    
    batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
    return x[batch_idx, sorted_indices]



def generate_greedy(model, prefix_ids, eos_id, max_gen_len, device):
    """Greedy INDIGO generation following Algorithm 1 in the paper.

    Args:
        model: IndigoTransformer instance
        prefix_ids: (1, prefix_len) tensor of context token ids (e.g. BOS + CDS + UTR5 tag)
        eos_id: token id for end-of-sequence
        max_gen_len: maximum number of tokens to generate
        device: torch device

    Returns:
        generated_ids: (1, gen_len) tensor of generated tokens in final l2r order
    """
    model.eval()
    prefix_ids = prefix_ids.to(device)
    prefix_len = prefix_ids.size(1)

    # Start with no generated tokens; R for just the prefix (l2r)
    gen_tokens = []  # list of token ids in generation order
    # R covers full sequence: prefix + generated so far
    seq_len = prefix_len
    R = torch.zeros(1, seq_len, seq_len, dtype=torch.long, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if i < j:
                R[0, i, j] = -1
            elif i > j:
                R[0, i, j] = 1

    input_ids = prefix_ids.clone()  # (1, seq_len)

    with torch.no_grad():
        for t in range(max_gen_len):
            H, _, word_logits = model(input_ids, R)

            # Predict next word from the last position's hidden state
            z = word_logits[0, -1, :].argmax().item()

            if z == eos_id:
                gen_tokens.append(z)
                break

            gen_tokens.append(z)

            if t == 0:
                # First generated token: just append, extend R
                new_R = torch.zeros(1, seq_len + 1, seq_len + 1, dtype=torch.long, device=device)
                new_R[0, :seq_len, :seq_len] = R[0]
                # New token is to the right of all prefix tokens
                new_R[0, :seq_len, seq_len] = -1
                new_R[0, seq_len, :seq_len] = 1
                R = new_R
            else:
                # Predict position for the new token
                H_gen = H[0, prefix_len:, :].unsqueeze(0)  # (1, n_gen, d)
                z_emb = model.get_embedding_matrix()[z]     # (d,)
                p_logits = model.position_logits(
                    H_gen, z_emb.unsqueeze(0),
                    bos_gen_idx=0,  # don't insert left of first gen token
                )
                p = p_logits[0].argmax().item()

                # Build R for generated tokens only, then reconstruct full R
                n_gen = len(gen_tokens) - 1  # tokens before this one
                R_gen = R[0, prefix_len:, prefix_len:]  # (n_gen, n_gen)
                R_gen = insert_relative_position_to_matrix(p, R_gen.unsqueeze(0)).squeeze(0)

                new_seq_len = seq_len + 1
                new_R = torch.zeros(1, new_seq_len, new_seq_len, dtype=torch.long, device=device)
                new_R[0, :prefix_len, :prefix_len] = R[0, :prefix_len, :prefix_len]
                new_R[0, prefix_len:, prefix_len:] = R_gen
                # Cross terms: all generated tokens are to the right of prefix
                new_R[0, :prefix_len, prefix_len:] = -1
                new_R[0, prefix_len:, :prefix_len] = 1
                R = new_R

            seq_len += 1
            input_ids = torch.cat([input_ids, torch.tensor([[z]], device=device)], dim=1)

    if len(gen_tokens) == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)

    gen_tensor = torch.tensor([gen_tokens], device=device)
    R_gen = R[0, prefix_len:, prefix_len:].unsqueeze(0)
    return restore_permutation(gen_tensor, R_gen)



def _build_l2r_R(n, device):
    """Build l2r relative position matrix for n tokens."""
    R = torch.zeros(n, n, dtype=torch.long, device=device)
    for i in range(n):
        for j in range(n):
            if i < j:
                R[i, j] = -1
            elif i > j:
                R[i, j] = 1
    return R


def _extend_full_R(R_full, R_gen_new, prefix_len, device):
    """Rebuild the full R matrix after extending the generated portion."""
    new_seq_len = prefix_len + R_gen_new.size(0)
    new_R = torch.zeros(new_seq_len, new_seq_len, dtype=torch.long, device=device)
    new_R[:prefix_len, :prefix_len] = R_full[:prefix_len, :prefix_len]
    new_R[prefix_len:, prefix_len:] = R_gen_new
    new_R[:prefix_len, prefix_len:] = -1  # prefix is left of all generated
    new_R[prefix_len:, :prefix_len] = 1   # generated is right of all prefix
    return new_R


def generate_beam_search(model, prefix_ids, eos_id, beam_size, max_gen_len, device):
    """Beam search INDIGO generation following the paper.

    Args:
        model: IndigoTransformer instance
        prefix_ids: (1, prefix_len) tensor of context token ids
        eos_id: token id for end-of-sequence
        beam_size: number of beams (B)
        max_gen_len: maximum number of tokens to generate
        device: torch device

    Returns:
        generated_ids: (1, gen_len) tensor of generated tokens in final l2r order
    """
    model.eval()
    prefix_ids = prefix_ids.to(device)
    prefix_len = prefix_ids.size(1)

    R_prefix = _build_l2r_R(prefix_len, device)

    # Each beam: (input_ids, R_full, R_gen, gen_tokens, log_score)
    # R_gen tracks only the generated portion for insert_relative_position_to_matrix
    init_R_full = torch.zeros(prefix_len, prefix_len, dtype=torch.long, device=device)
    init_R_full[:] = R_prefix
    beams = [(prefix_ids.squeeze(0), init_R_full, None, [], 0.0)]
    completed = []

    with torch.no_grad():
        for t in range(max_gen_len):
            all_candidates = []

            for input_seq, R_full, R_gen, gen_tokens, score in beams:
                cur_len = input_seq.size(0)
                H, _, word_logits = model(
                    input_seq.unsqueeze(0),
                    R_full.unsqueeze(0),
                )

                # Word prediction from last hidden state
                log_probs_word = F.log_softmax(word_logits[0, -1, :], dim=-1)
                top_word_scores, top_word_ids = torch.topk(log_probs_word, beam_size)

                for w_score, z_id in zip(top_word_scores, top_word_ids):
                    z = z_id.item()
                    new_gen_tokens = gen_tokens + [z]
                    new_input = torch.cat([input_seq, z_id.unsqueeze(0)])

                    if z == eos_id:
                        # Completed beam — no position prediction needed
                        # Extend R with EOS to the right of everything
                        if R_gen is not None:
                            new_R_gen = insert_relative_position_to_matrix(
                                len(gen_tokens) + len(gen_tokens) - 1,  # rightmost position
                                R_gen.unsqueeze(0)
                            ).squeeze(0)
                        else:
                            new_R_gen = torch.zeros(1, 1, dtype=torch.long, device=device)
                        new_R_full = _extend_full_R(R_full, new_R_gen, prefix_len, device)
                        completed.append((new_input, new_R_full, new_R_gen, new_gen_tokens, score + w_score.item()))
                        continue

                    if t == 0:
                        # First generated token: R_gen is just (1,1) zeros
                        new_R_gen = torch.zeros(1, 1, dtype=torch.long, device=device)
                        new_R_full = _extend_full_R(R_full, new_R_gen, prefix_len, device)
                        all_candidates.append((new_input, new_R_full, new_R_gen, new_gen_tokens, score + w_score.item()))
                    else:
                        # Position prediction
                        H_gen = H[0, prefix_len:, :].unsqueeze(0)
                        z_emb = model.get_embedding_matrix()[z]
                        p_logits = model.position_logits(
                            H_gen, z_emb.unsqueeze(0),
                            bos_gen_idx=0,
                        )
                        log_probs_pos = F.log_softmax(p_logits[0], dim=-1)
                        top_pos_scores, top_pos_ids = torch.topk(log_probs_pos, min(beam_size, log_probs_pos.size(0)))

                        for p_score, p_id in zip(top_pos_scores, top_pos_ids):
                            p = p_id.item()
                            new_R_gen = insert_relative_position_to_matrix(
                                p, R_gen.unsqueeze(0)
                            ).squeeze(0)
                            new_R_full = _extend_full_R(R_full, new_R_gen, prefix_len, device)
                            all_candidates.append((
                                new_input, new_R_full, new_R_gen, new_gen_tokens,
                                score + w_score.item() + p_score.item()
                            ))

            if not all_candidates:
                break

            # Keep top beams by score, normalized by length
            all_candidates.sort(key=lambda x: x[4] / max(len(x[3]), 1), reverse=True)
            beams = all_candidates[:beam_size]

    # Choose best from completed (or beams if none completed)
    final = completed if completed else beams
    best = max(final, key=lambda x: x[4] / max(len(x[3]), 1))
    _, _, R_gen_best, gen_tokens_best, _ = best

    if len(gen_tokens_best) == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)

    gen_tensor = torch.tensor([gen_tokens_best], device=device)
    return restore_permutation(gen_tensor, R_gen_best.unsqueeze(0))


class IndigoTransformer(nn.Module):
    """Full INDIGO model: encoder + word head + position head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = IndigoEncoder(config)
        # Paper Eq. 8: p_word = softmax(h^T F W^T), F projects hidden state,
        # W is the token embedding matrix (tied weights)
        self.word_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.position_head_state_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.position_head_left_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.position_head_right_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, input_ids, R, attention_mask=None):
        H, R = self.encoder(input_ids, R, attention_mask)
        W = self.get_embedding_matrix()  # (vocab_size, d_model)
        word_logits = torch.matmul(self.word_proj(H), W.T)  # (batch, seq_len, vocab_size)
        return H, R, word_logits

    def position_logits(self, H, z_embedding, bos_gen_idx=None, eos_gen_idx=None):
        """Compute position logits for inserting the next token.

        H: (batch, n, d_model) hidden states of n already-generated tokens
        z_embedding: (batch, d_model) embedding of the token to be inserted
        bos_gen_idx: generation index of the BOS/start boundary token (mask left-of)
        eos_gen_idx: generation index of the EOS/end boundary token (mask right-of)

        Returns logits of shape (batch, 2n): first n = left-of, last n = right-of
        """
        left = self.position_head_left_proj(H)
        right = self.position_head_right_proj(H)
        query = (self.position_head_state_proj(H[:, -1:, :]) + z_embedding.unsqueeze(1))
        keys = torch.cat([left, right], dim=1).transpose(-1, -2)
        pos_logits = torch.matmul(query, keys).squeeze(1)

        # Paper below Eq. 9: mask invalid boundary positions
        n = H.size(1)
        if bos_gen_idx is not None:
            # Mask left-of-BOS (position = bos_gen_idx in left half)
            pos_logits[:, bos_gen_idx] = float('-inf')
        if eos_gen_idx is not None:
            # Mask right-of-EOS (position = n + eos_gen_idx in right half)
            pos_logits[:, n + eos_gen_idx] = float('-inf')

        return pos_logits

    def get_embedding_matrix(self):
        return self.encoder.embedding_layer.token_embedding.weight