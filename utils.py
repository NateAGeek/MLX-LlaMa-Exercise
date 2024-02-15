import mlx.core as mx
import mlx.nn as nn
from sentencepiece import SentencePieceProcessor

def generate(model: nn.Module, tokenizer: SentencePieceProcessor, prompt: str, temp=0.0, max_tokens=2048, write_every=10):
    print("------")
    print(prompt)
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    skip = 0
    tokens = []
    for token in model.generate(x, temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)

        if len(tokens) >= max_tokens:
            break

        elif (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")