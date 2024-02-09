import mlx.core as mx
import mlx.nn as nn
from attention import AttentionLayer
from swiglu import SwiGLU

class DecoderLayer(nn.Module):
    def __init__(self, token_dimensions: int, mlp_dimensions:int, number_of_heads: int):
        super().__init__()
        self.embedding_normalization = nn.RMSNorm(token_dimensions)

        self.self_attention = AttentionLayer(token_dimensions, number_of_heads)

        self.attention_normalization = nn.RMSNorm(token_dimensions)

        self.linear_gate = nn.Linear(token_dimensions, mlp_dimensions, bias=False)
        self.linear_gate_feedforward = nn.Linear(token_dimensions, mlp_dimensions, bias=False)
        self.linear_feedforward = nn.Linear(mlp_dimensions, token_dimensions, bias=False)

    def __call__(self, x, mask=None, key_value_cache=None):
        
        attention_output, key_value_cache = self.self_attention(
            self.embedding_normalization(x),
            mask=mask,
            key_value_cache=key_value_cache
        )

        attention_output = x + attention_output

        attention_normalization = self.attention_normalization(attention_output)
        
        attention_output = self.linear_feedforward(nn.silu(self.linear_gate(attention_normalization)) * self.linear_gate_feedforward(attention_normalization))
        
        output = x + attention_output

        return output, key_value_cache
    