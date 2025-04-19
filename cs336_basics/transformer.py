import torch
import torch.nn as nn
from einops import einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.exp(x - x_max) / torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)

def Swish(x: torch.Tensor) -> torch.Tensor:
    # Swish(x) = x * sigmoid(x)
    return x * torch.sigmoid(x)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''
    Args:
        Q: (batch, seq_len, d_k)
        K: (batch, seq_len, d_k)
        V: (batch, seq_len, d_v)
        mask: (batch, seq_len, seq_len) - optional mask, 1 for valid positions, 0 for masked ones
    Returns:
        Output tensor: (batch, seq_len, d_v)
    '''

    d_k = Q.size(-1)
    qk = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    if mask is not None:
        qk = qk.masked_fill(mask == 0, float('-inf'))

    attn = softmax(qk / d_k ** 0.5, dim=-1)
    output = einsum(attn, V, '... queries keys, ... keys d_v -> ... queries d_v')

    return output


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

        std = (2.0 / (in_features + out_features))**0.5
        min_val = -3.0 * std
        max_val = 3.0 * std
        self.init_parameters(std, min_val, max_val)

    def init_parameters(self, std: float = 0.02, min_val: float = -1, max_val: float = 1):
        # Initialize the weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=min_val, b=max_val)

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

    def init_parameters(self, std: float = 1, min_val: float = -3, max_val: float = 3):
        # Initialize the weights with a truncated normal distribution
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=std, a=min_val, b=max_val)

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
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
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

        return x_norm.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # Parameters initialization
        self.W_1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W_2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.W_3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        self.init_parameters()

    def init_parameters(self, std: float = 0.02, min_val: float = -1, max_val: float = 1):
        std_w1_w2 = (2 / (self.d_model + self.d_ff)) ** 0.5
        std_w3 = (2 / (self.d_ff + self.d_model)) ** 0.5

        nn.init.trunc_normal_(self.W_1, mean=0, std=std_w1_w2, a=-3*std_w1_w2, b=3*std_w1_w2)
        nn.init.trunc_normal_(self.W_2, mean=0, std=std_w1_w2, a=-3*std_w1_w2, b=3*std_w1_w2)
        nn.init.trunc_normal_(self.W_3, mean=0, std=std_w3, a=-3*std_w3, b=3*std_w3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        a = Swish(einsum(x, self.W_1, "... seq d_model, d_ff d_model -> ... seq d_ff"))
        b = einsum(x, self.W_3, "... seq d_model, d_ff d_model -> ... seq d_ff")

        swiglu_output = a * b  # (..., seq, d_ff)

        output = einsum(swiglu_output, self.W_2, "... seq d_ff, d_model d_ff -> ... seq d_model")

        return output


class RotaryPositionalEmbedding(nn.Module):
    """
    Apply RoPE embeddings to input tensor x using precomputed rotations.
    """
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

        position = torch.arange(max_seq_len, device=device).float().unsqueeze(1)  # (max_seq_len, 1)
        dim_pair = torch.arange(0, d_k, 2, device=device).float()  # (d_k/2)
        inv_freq = 1.0 / (theta ** (dim_pair / d_k))  # (d_k/2)
        angles = position * inv_freq # (max_seq_len, d_k/2)

        # Precompute sin and cos tensors
        self.register_buffer("cos", torch.cos(angles), persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("sin", torch.sin(angles), persistent=False)  # (max_seq_len, d_k/2)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE embeddings to input tensor x using precomputed rotations.

        Args:
            x (torch.Tensor): Input embeddings (..., seq_len, d_k)
            token_positions (torch.Tensor): Positions of tokens (..., seq_len)

        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as input x
        """
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)

        # Split dimensions into even and odd pairs
        x1, x2 = x[..., 0::2], x[..., 1::2]  # both (..., seq_len, d_k/2)

        # Apply rotation explicitly
        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd = x1 * sin + x2 * cos

        # Combine back into original tensor shape
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

        return x_rotated


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        '''
        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters

        Folllowing Vaswani et al. [2017], set dk = dv = dmodel /h
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass
