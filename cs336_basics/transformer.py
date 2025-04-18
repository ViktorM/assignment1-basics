import torch
import torch.nn as nn
import einops
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.init_parameters()

    def init_parameters(self, std: float = 0.02):
        # Initialize the weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform a matrix multiplication of the input with the transposed weights
        # This is equivalent to a dot product between the input and each row of the weight matrix
        # This is a linear transformation
        # x: (batch, in_feat)
        # self.weight: (out_feat, in_feat)
        # y (batch, out_feat)
        # y = x @ self.weight.T
        y = einsum(x, self.weight, 'batch in_feat, out_feat in_feat -> batch out_feat')

        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        '''
        '''
        pass

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass
