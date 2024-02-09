import argparse
from itertools import starmap

import numpy as np
import torch

def map_torch_to_mlx(key, value):
    if "tok_embedding" in key:
        key = "token_embedding.weight"

    elif "norm.weight" == key:
        key = key.replace("norm", "normalization")

    elif "norm" in key:
        key = key.replace("attention_norm", "attention_normalization").replace("ffn_norm", "embedding_normalization")

    elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
        key = key.replace("wq", "query_weights")
        key = key.replace("wk", "key_weights")
        key = key.replace("wv", "value_weights")
        key = key.replace("wo", "linear_output")

    elif "w1" in key or "w2" in key or "w3" in key:
        # The FFN is a separate submodule in PyTorch
        key = key.replace("feed_forward.w1", "linear_gate")
        key = key.replace("feed_forward.w2", "linear_gate_feedforward")
        key = key.replace("feed_forward.w3", "linear_feedforward")

    elif "output" in key:
        key = key.replace("output", "linear_feedforward")

    elif "rope" in key:
        return None, None

    value = value.to(getattr(torch, "float16"))
    print(f"MODEL LAYER: {key} -> LAYER SHAPE {value.shape}")
    return key, value.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Llama weights to MLX")
    parser.add_argument("torch_weights")
    parser.add_argument("output_file")
    args = parser.parse_args()

    state = torch.load(args.torch_weights)
    np.savez(
        args.output_file,
        **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
    )