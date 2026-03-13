import torch
import torch.nn as nn
import torch.nn.functional as F


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
        R = self.relative_positional_embedding.weight[r] # shape: (seq_len, seq_len, dmodel)

        S = torch.einsum('ik,jk,ijk->ijk', query, key, R)
        # instead of S[i, j] = Q[i] * K[j] in ordinary attention, we introduce relative position bias
        # S[i, j] = Q[i] * (K[j] + R[i, j])

        attention_weights = torch.softmax(((1 / torch.sqrt(self.dmodel)) * S), dim=1)
        attention_output = torch.matmul(attention_weights, value)
        # for now I've implemented a plain formula from the paper, without considering heads > 1 and batch_size > 1
        # I also don't yet know, how to use attention backend for this

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output

class DecodingLayer(nn.Module):

    def __init__(self, dmodel, vocab_size):
        super(DecodingLayer, self).__init__()
        
        self.proj = nn.Linear(dmodel, dmodel, bias=False)
        self.dmodel = dmodel
        self.vocab_size = vocab_size

    def forward(self, H, embedding_matrix):
        pass

class IndigoTransformer(nn.Module):

    def __init__(self):
        super(IndigoTransformer, self).__init__()
        pass

    def forward(self):
        pass

def indigo_loss_fn():
    pass


def train():
    pass

def generate():
    pass


def main():
    pass


if __name__ == "__main__":
    main()