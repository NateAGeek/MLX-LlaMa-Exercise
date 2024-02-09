import mlx.core as mx
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor

from llama import LLaMa


model_path = "./llama-7B.mlx.npz"

vocabulary_size = 32000
mlp_dimensions = 11008
token_dimensions = 4096
number_of_heads = 32
number_of_decoder_layers = 32

print("Loading model with the following parameters:")
print(f"Vocabulary size: {vocabulary_size}")
print(f"MLP dimensions: {mlp_dimensions}")
print(f"Token dimensions: {token_dimensions}")
print(f"Number of heads: {number_of_heads}")
print(f"Number of Decoder Layers: {number_of_decoder_layers}")

llama_7B_model = LLaMa(
    number_of_decoder_layers=number_of_decoder_layers,
    number_of_heads=number_of_heads,
    mlp_dimensions=mlp_dimensions,
    token_dimensions=token_dimensions,
    vocabulary_size=vocabulary_size
)

llama_7B_model.update(tree_unflatten(list(mx.load(model_path).items())))

llama_7B_model_tokenizer = SentencePieceProcessor(model_file=str("./tokenizer.model"))

def generate(model, tokenizer, prompt, temp=0.7, max_tokens=2048, write_every=10):
    input("Press enter to start generation")
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

print("Model loaded successfully!")

generate(llama_7B_model, llama_7B_model_tokenizer, "In the beginning the Universe was created.", temp=0.7, max_tokens=2048, write_every=1)
