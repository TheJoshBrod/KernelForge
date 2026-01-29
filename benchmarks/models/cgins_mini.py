import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG = {
    "vocab_size": 1000,
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_classes": 10,
    "dropout": 0.1,
}


class CGinSMini(nn.Module):
    """Small model that exercises common ops used by CGinS."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq]
        x = self.emb(input_ids)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return self.fc2(x)


def build_model(config: dict | None = None) -> CGinSMini:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    return CGinSMini(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    )


def load_weights(path: str, device: str = "cpu") -> CGinSMini:
    payload = torch.load(path, map_location=device, weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload:
        config = payload.get("config", DEFAULT_CONFIG)
        state_dict = payload["state_dict"]
    else:
        config = DEFAULT_CONFIG
        state_dict = payload

    model = build_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_example_input(batch: int = 8, seq: int = 16, vocab_size: int = DEFAULT_CONFIG["vocab_size"]) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch, seq))
