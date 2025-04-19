import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        Returns:
            None
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
            )

        self.init_parameters()

    def init_parameters(self, std: float = 0.02):
        # Initialize the weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight, std=std)

    def load_parameters(self, weights: torch.Tensor):
        # Ensure shape matches
        assert weights.shape == self.weight.shape, \
            f"Expected weights shape {self.weight.shape}, but got {weights.shape}."

        with torch.no_grad():
            self.weight.copy_(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform a matrix multiplication of the input with the transposed weights
        # This is equivalent to a dot product between the input and each row of the weight matrix
        # This is a linear transformation
        # x: (batch, in_feat)
        # self.weight: (out_feat, in_feat)
        # y (batch, out_feat)
        # y = x @ self.weights.T
        y = einsum(x, self.weight, '... in_feat, out_feat in_feat -> ... out_feat')

        return y


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        * initialize your embedding matrix as a nn.Parameter
        * store the embedding matrix with the d_model being the final dimension
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
            )

        self.init_parameters()

    def init_parameters(self, std: float = 0.02):
        # Initialize the weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.embedding, std=std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        The forward method should select the embedding vector for each token ID 
        by indexing into an embedding matrix of shape (vocab_size, d_model) using 
        a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        Args:
            token_ids: (batch, seq_len)
        Returns:
            embedding: (batch, seq_len, d_model)
        '''
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Construct the RMSNorm module. This function should accept the following parameters:

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Should we init it?
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape. Remember to upcast your input to 
        torch.float32 before performing the normalization (and later downcast to 
        the original dtype), as described above.
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x_normed: (batch, seq_len, d_model)
        '''
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.g
        x.norm = x_norm.to(in_dtype)

        return x_norm


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
