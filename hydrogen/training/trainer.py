"""Unified Trainer with explicit train / val split support."""

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from hydrogen.backbones import get_backbone


def get_model(
    backbone: str = "physicsnemo_fno",
    in_channels: int = 3,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    BackboneClass = get_backbone(backbone)
    return BackboneClass(in_channels=in_channels, out_channels=out_channels, **kwargs)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader: Optional[Any] = None,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    physics_loss_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)

            loss = torch.nn.functional.mse_loss(pred, y)

            if physics_loss_fn is not None:
                loss = loss + physics_loss_fn(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    pred = model(x)
                    val_loss += torch.nn.functional.mse_loss(pred, y).item()
            avg_val = val_loss / len(val_loader)
            history["val_loss"].append(avg_val)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.6f}")

    return {"model": model, "history": history}
