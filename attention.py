import mlx.core as mx
import mlx.nn as nn
import math

class AttentionLayer(nn.Module):
    def __init__(self, token_dimensions: int, number_of_heads: int ):
        super().__init__()
        self.number_of_heads = number_of_heads
        self.scale = token_dimensions**-0.5

        # The use of RoPE allows the attention layer to encode positional information
        # for the query and key, as value is not needed for the positional encoding.
        self.positional_rope = nn.RoPE(token_dimensions // number_of_heads, traditional=True)

        self.query_weights = nn.Linear(token_dimensions, token_dimensions, bias=False)
        self.key_weights = nn.Linear(token_dimensions, token_dimensions, bias=False)
        self.value_weights = nn.Linear(token_dimensions, token_dimensions, bias=False)

        self.linear_output = nn.Linear(token_dimensions, token_dimensions, bias=False)
        

    def __call__(self, x, mask=None, key_value_cache=None):
        projected_queries = self.query_weights(x)
        projected_keys = self.key_weights(x)
        projected_values = self.value_weights(x)

        number_of_heads = self.number_of_heads
        batch_size, sequence_length, _ = projected_queries.shape

        # Reshape the queries, keys, and values to have the shape
        # (batch_size, sequence_length, num_heads, token_dimensions // num_heads)
        # Then transpose them to have the shape (batch_size, num_heads, sequence_length, token_dimensions // num_heads)
        projected_queries = projected_queries.reshape(
            batch_size,
            sequence_length,
            number_of_heads,
            -1
        ).transpose(0, 2, 1, 3)
        projected_keys    = projected_keys.reshape(
            batch_size,
            sequence_length,
            number_of_heads,
            -1
        ).transpose(0, 2, 1, 3)
        projected_values  = projected_values.reshape(
            batch_size,
            sequence_length,
            number_of_heads,
            -1
        ).transpose(0, 2, 1, 3)

        # If we have a cache, preform the KV Cache technique
        # TODO: Better describe the KV Cache technique
        if key_value_cache is not None:
            projected_key_cache, projected_value_cache = key_value_cache
            projected_queries = self.positional_rope(
                projected_queries,
                offset=projected_key_cache.shape[2]
            )
            projected_keys = self.positional_rope(
                projected_keys,
                offset=projected_key_cache.shape[2]
            )

            projected_keys = mx.concatenate([
                projected_key_cache,
                projected_keys
            ], axis=2)

            projected_values = mx.concatenate([
                projected_value_cache,
                projected_values
            ], axis=2)
        else:
            # Encode the positional information for the queries and keys
            projected_queries = self.positional_rope(projected_queries)
            projected_keys = self.positional_rope(projected_keys)

        scores = (projected_queries * self.scale) @ projected_keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        
        attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attention_output = (attention @ projected_values).transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            -1
        )
        return self.linear_output(attention_output), (projected_keys, projected_values)
     