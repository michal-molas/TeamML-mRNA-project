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


class IndigoTransformer(nn.Module):
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

    def get_embedding_matrix(self):
        return self.encoder.embedding_layer.token_embedding.weight

#### TODO

def insert_relative_position_to_matrix(p, R):
    
    # extends relative positon matrix R with a new position p
    # p is an integer from range [0, 2n-1] 
    # R is an (n x n) matrix

    n = R.shape[-1] # length of the previously generated sequence
    neighbour_token = p % n 

    r = R[neighbour_token]
    if p < n:
        r[neighbour_token] = 1
    else:
        r[neighbour_token] = -1

    return torch.cat([torch.cat([R, r.view(n, 1)], dim=-1), 
               torch.cat([(-r).view(1, n), torch.zeros(1,1)], dim=-1)], dim=-2)


def restore_permutation(x, R):

    # permutes the sequence according to the relative position matrix

    R = R.copy()
    R[R == -1] = 0
    positions = torch.sum(R, dim=-1)
    return x[positions]



def generate_greedy(model, cds_seq):
    # inputs a CDS sequence
    # outputs generated UTR sequence
    # generates greedily - at each timestep chooses the next token with the highest probability
    # then its relative position is also the one with the highest probability

    utr_seq = start_token
    R = torch.zeros(1,)

    while not eod: 
        H, R = model.encoder(cds_seq, utr_seq, R)
        Z = model.word_decoder(H, R) # next token probabilities
        z = torch.argmax(Z) # next token 
        P = model.position_decoder(H, R, z) # relative position probabilities
        p = torch.argmax(P) # relative position
        R = insert_relative_position_to_matrix(p, R)

    utr_seq = restore_permutation(utr_seq, R)
    return utr_seq



def generate_beam_search(model, input_ids, beam_size, max_len):
    
    # inputs a CDS sequence
    # outputs generated UTR sequence
    # uses beam search as described in the indigo paper

    # Each beam: (sequence, relative position matrix, score)
    beams = [(input_ids, torch.zeros(1,), 0.0)]
    completed = []

    for t in range(max_len):

        all_candidates = []

        for seq, R, score in beams:

            if seq[0, -1].item() == eod_token:
                completed.append((seq, R, score))
                continue

            H, R = model.encoder(input_ids, seq, R)
            Z = model.word_decoder(H, R) # next token probabilities

            top_probs, top_tokens = torch.topk(Z, beam_size, dim=-1)

            for prob, z in zip(top_probs, top_tokens):
                new_seq = torch.cat([seq, z], dim=-1)
                new_score = score + prob

                P = model.position_decoder(H, R, z)

                top_position_probs, top_positions = torch.topk(P, beam_size, dim=-1)

                for prob_position, p in zip(top_probs, top_tokens):
                    new_R = insert_relative_position_to_matrix(p, R)
                    new_score = new_score + prob_position
                    all_candidates.append((new_seq, new_R, new_score))

        beams = all_candidates[:beam_size]

    def normalize_score(seq, score):
        length = seq.shape[-1]
        return score / length

    # Choose best sequence
    final_candidates = completed if len(completed) > 0 else beams
    best_seq, best_R, best_score = max(
        final_candidates,
        key=lambda x: normalize_score(x[0], x[1])
    )

    best_seq = restore_permutation(best_seq, best_R)

    return best_seq

#### TODO end
