from typing import List, Union
import torch
from src.first_level.models.nn.activation import Activation


class FeedForward(torch.nn.Module):
    """
    This `Module` is a feed-forward neural network, just a sequence of `Linear` layers with
    activation functions and dropout in between.

    Parameters
    ----------
    input_size : `int`, required
        The dimensionality of the input.  We assume the input has shape `(batch_size, input_size)`.
    num_layers : `int`, required
        The number of `Linear` layers to apply to the input.
    hidden_sizes : `Union[int, List[int]]`, required
        The output dimension of each of the `Linear` layers.  If this is a single `int`, we use
        it for all `Linear` layers.  If it is a `List[int]`, `len(hidden_dims)` must be
        `num_layers`.
    activations : `Union[Activation, List[Activation]]`, required
        The activation function to use after each `Linear` layer.  If this is a single function,
        we use it after all `Linear` layers.  If it is a `List[Activation]`,
        `len(activations)` must be `num_layers`. Activation must have torch.nn.Module type.
    dropout : `Union[float, List[float]]`, optional (default = `0.0`)
        If given, we will apply this amount of dropout after each layer.  Semantics of `float`
        versus `List[float]` is the same as with other parameters.
    output_size : `int`, optional (default = `None`)
        Output size for Module. If None last dimension of hidden_dims is used.

    Examples
    --------
    ```python
    FeedForward(124, 2, [64, 32], Activation("relu"), 0.2)
    >>> FeedForward(
    >>>   (_activations): ModuleList(
    >>>     (0): Activation(relu)
    >>>     (1): Activation(relu)
    >>>   )
    >>>   (_linear_layers): ModuleList(
    >>>     (0): Linear(in_features=124, out_features=64, bias=True)
    >>>     (1): Linear(in_features=64, out_features=32, bias=True)
    >>>   )
    >>>   (_dropout): ModuleList(
    >>>     (0): Dropout(p=0.2, inplace=False)
    >>>     (1): Dropout(p=0.2, inplace=False)
    >>>   )
    >>>   (_output): Linear(in_features=32, out_features=32, bias=True)
    >>> )
    ```
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_sizes: Union[int, List[int]],
        activations: Union[str, Activation, List[Activation]],
        dropout: Union[float, List[float]] = 0.0,
        output_size: int = None,
    ) -> None:
        super().__init__()
        if isinstance(activations, str):
            activations = Activation(activations)
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        if len(hidden_sizes) != num_layers:
            raise ValueError(f"len(hidden_dims) ({len(hidden_sizes)}) != num_layers ({num_layers})")
        if len(activations) != num_layers:
            raise ValueError(f"len(activations) ({len(activations)}) != num_layers ({num_layers})")
        if len(dropout) != num_layers:
            raise ValueError(f"len(dropout) ({len(dropout)}) != num_layers ({num_layers})")
        self._input_size = input_size
        self._output_size = hidden_sizes[-1]
        # Modules
        self._linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer_input, layer_output)
                for layer_input, layer_output in zip([input_size] + hidden_sizes[:-1], hidden_sizes)
            ]
        )
        self._activations = torch.nn.ModuleList(activations)
        self._dropout = torch.nn.ModuleList([torch.nn.Dropout(prob) for prob in dropout])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            x = dropout(activation(layer(x)))
        return x

    def get_input_size(self) -> int:
        return self._input_size

    def get_output_size(self) -> int:
        return self._output_size
