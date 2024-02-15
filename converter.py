import argparse
from itertools import starmap

import mlx.core as mx
import torch

def map_torch_to_mlx(key, value):
    if key == "tok_embeddings.weight":
        key = "token_embedding.weight"
    
    if key == "norm.weight":
        key = "normalization.weight"
    
    if key == "output.weight":
        key = "linear_feedforward.weight"

    if "layers" in key:
        key = key.replace("layers.", "decoder_layers.")
    
    if ".attention." in key:
        key = key.replace(".attention.", ".self_attention.")
    
    if ".ffn_norm." in key:
        key = key.replace(".ffn_norm.", ".attention_normalization.")

    if ".attention_norm." in key:
        key = key.replace(".attention_norm.", ".embedding_normalization.")
    
    if "feed_forward.w1" in key:
        key = key.replace("feed_forward.w1", "linear_gate")
    
    if "feed_forward.w2" in key:
        key = key.replace("feed_forward.w2", "linear_feedforward")
    
    if "feed_forward.w3" in key:
        key = key.replace("feed_forward.w3", "linear_gate_feedforward")

    if "wq" in key:
        key = key.replace("wq", "query_weights")
    
    if "wk" in key:
        key = key.replace("wk", "key_weights")
    
    if "wv" in key:
        key = key.replace("wv", "value_weights")

    if "wo" in key:
        key = key.replace("wo", "output_weights")
    
    #Skip rope, as it is not used in the model
    if "rope" in key:
        return None, None

    a = value.to(getattr(torch, "float16"))
    value = mx.array(a.numpy(), getattr(mx, "float16"))
    print(f"MODEL LAYER: {key} -> LAYER SHAPE {value.shape}")
    return key, value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument("torch_weights")
    parser.add_argument("output_file")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    mx.savez(str(args.output_file), **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None})
    