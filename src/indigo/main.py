import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses 



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

        self.relative_positional_embedding = nn.Linear(3, dmodel, bias=False) 
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

        r += 1 # {-1, 0, 1} to valid indices {0, 1, 2}
        R = self.relative_positional_embedding.weight[r] # shape: (batch, seq_len, seq_len, dmodel)

        q_dot_k = torch.matmul(query, key.transpose(-2, -1))
        q_dot_r = torch.einsum('bhid,bijd->bhij', query, R)
        S = q_dot_k + q_dot_r
        # instead of S[i, j] = Q[i] * K[j] in ordinary attention, we introduce relative position bias
        # S[i, j] = Q[i] * (K[j] + R[i, j])
        assert S.shape == (batch, self.heads, seq_len, seq_len)

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

                for prob_position, p in zip(top_position_probs, top_positions):
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

    # TODO

def train():
    pass

    # TODO
    # i dont understand training and loss function in the indigo paper


def main():
    pass

    # TODO


if __name__ == "__main__":
    main()