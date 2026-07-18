"""Highly modular unified trainer.

Respects as many strategy fields as possible.
"""

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


def get_optimizer(model: nn.Module, strategy: dict):
    """Create optimizer from strategy config."""
    opt_name = strategy.get("optimizer", "AdamW").lower()
    lr = strategy.get("learning_rate", 0.001)
    weight_decay = strategy.get("weight_decay", 1e-4)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = strategy.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        # Default fallback
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, strategy: dict, epochs: int):
    """Create learning rate scheduler from strategy."""
    scheduler_name = strategy.get("scheduler", "CosineAnnealingLR").lower()

    if scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "steplr":
        step_size = strategy.get("step_size", max(1, epochs // 4))
        gamma = strategy.get("gamma", 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader: Optional[Any] = None,
    strategy: dict = None,
    epochs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    physics_loss_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    if strategy is None:
        strategy = {}

    model = model.to(device)
    optimizer = get_optimizer(model, strategy)
    scheduler = get_scheduler(optimizer, strategy, epochs)

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

        # Validation
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

        # Step scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_train)
        else:
            scheduler.step()

    return {"model": model, "history": history}
