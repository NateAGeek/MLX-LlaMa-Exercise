import mlx.core as mx
from sentencepiece import SentencePieceProcessor
from llama.llama import LLaMa
import utils

model_path = "./models/llama-7B/llama-7B.mlx.npz"

mx.random.seed(0)
vocabulary_size = 32000
mlp_dimensions = 11008
token_dimensions = 4096
number_of_heads = 32
number_of_decoder_layers = 32
temperature = 0.0

llama_7B_model = LLaMa(
    number_of_decoder_layers=number_of_decoder_layers,
    number_of_heads=number_of_heads,
    mlp_dimensions=mlp_dimensions,
    token_dimensions=token_dimensions,
    vocabulary_size=vocabulary_size
)   

print("Loading model with the following parameters:")
print(f"Vocabulary size: {vocabulary_size}")
print(f"MLP dimensions: {mlp_dimensions}")
print(f"Token dimensions: {token_dimensions}")
print(f"Number of heads: {number_of_heads}")
print(f"Number of Decoder Layers: {number_of_decoder_layers}")

llama_7B_model.load_weights(model_path)
llama_7B_model_tokenizer = SentencePieceProcessor(model_file=str("./models/llama-7B/tokenizer.model"))

utils.generate(
    llama_7B_model,
    llama_7B_model_tokenizer,
    "In the beginning the Universe was created.",
    temp=temperature,
    max_tokens=2048,
    write_every=1
)
