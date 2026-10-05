"""TinyFFN architecture and forward function for Safetensors + PyTorch."""

import torch

from ..common import Config


class TinyFFN(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(config.input_dim, config.hidden_dim)
        self.fc2 = torch.nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


MODEL_REGISTRY: dict[str, type[TinyFFN]] = {"tiny_ffn": TinyFFN}
