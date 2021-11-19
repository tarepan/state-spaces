""" Implementation of FFN block in the style of Transformers """

from torch import nn
from src.models.sequence.base import SequenceModule
from src.models.nn import LinearActivation
a
class FF(SequenceModule):
    def __init__(self, d_input, expand=2, d_output=None, transposed=False, activation='gelu', initializer=None, dropout=0.0):
        """
        Args:
            activation: Activation between FC1 and FC2
        """
        super().__init__()
        self.d_output = d_input if d_output is None else d_output
        self.transposed = transposed
        d_inner = expand * d_input

        linear1 = LinearActivation(
            d_input, d_inner,
            transposed=transposed,
            activation=activation,
            initializer=initializer,
            activate=True,
        )
        dropout_cls = nn.Dropout2d if self.transposed else nn.Dropout
        drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()

        linear2 = LinearActivation(
            d_inner, self.d_output,
            transposed=transposed,
            activation=None,
            initializer=initializer,
            activate=False,
        )

        # FC-Activation-Do-FC
        self.ff = nn.Sequential(
            linear1,
            drop,
            linear2,
        )

    def forward(self, x, *args, **kwargs):
        return self.ff(x), None

    def step(self, x, state):
        # x: [batch, d_input]
        if self.transposed:
            # expects: [batch, d_input, seq_len]
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state
        else:
            return self.ff(x), state

