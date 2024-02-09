import mlx.core as mx
import mlx.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, token_dimensions: int, mlp_dimensions: int):
        super().__init__()



    def __call__(self, y):

        a = self.linear_gate(y)
        b = self.linear_gate_feedforward(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear_feedforward(y)

        return y
