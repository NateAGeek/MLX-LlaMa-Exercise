import mlx.core as mx
import mlx.nn as nn
from llama.decoder import DecoderLayer

class LLaMa(nn.Module):
    def __init__(self, number_of_decoder_layers:int, vocabulary_size:int, token_dimensions:int, mlp_dimensions:int, number_of_heads:int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocabulary_size, token_dimensions)

        self.decoder_layers = [
            DecoderLayer(token_dimensions, mlp_dimensions, number_of_heads)
            for _ in range(number_of_decoder_layers)
        ]

        self.normalization = nn.RMSNorm(token_dimensions)
        self.linear_feedforward = nn.Linear(token_dimensions, vocabulary_size, bias=False)

    def __call__(self, x, mask=None, key_value_cache=None):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)

        token_embedding = self.token_embedding(x)

        for decoder_layer in self.decoder_layers:
            token_embedding, key_value_cache = decoder_layer(token_embedding, mask, key_value_cache)
        
        attention_normalization = self.normalization(token_embedding)
        output = self.linear_feedforward(attention_normalization)
        return output
    
    def generate(self, x, temp=0.0):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))
            
        generation_key_value_cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)

        # First we process the prompt x the same way as in __call__ but
        # save the caches in cache
        token_embedding = self.token_embedding(x)
        for layer in self.decoder_layers:
            token_embedding, key_value_cache = layer(token_embedding, mask)
            generation_key_value_cache.append(key_value_cache)

        attention_normalization = self.normalization(token_embedding)
        y = self.linear_feedforward(attention_normalization[:, -1])  # <--- we only care about the last logits
                                     #      that generate the next token
        y = sample(y)

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.token_embedding(x)
            for i in range(len(generation_key_value_cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, generation_key_value_cache[i] = self.decoder_layers[i](x, mask=None, key_value_cache=generation_key_value_cache[i])
            x = self.normalization(x)
            y = self.linear_feedforward(x[:, -1])
            y = sample(y)

            yield y
