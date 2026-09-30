import math
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from matplotlib import pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from .logging_utils import make_next_indexed_log_filename


def fit_with_early_stopping(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    lr_max: float = 1e-2,
    lr_min: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 20,
    patience: int = 3,
    shuffle_seed: Optional[int] = None,
    verbose: bool = False,
    logging: bool = False,
    log_dir: Optional[str] = None,
    log_filename: Optional[str] = None,
) -> int:
    """
    Trains a calibration head with per-epoch shuffling and early stopping.

    Mini-batch AdamW with a cosine schedule spanning ``max_epochs``. After
    every epoch the data loss on the validation set is measured; training
    stops once it has not improved for ``patience`` epochs, and the weights
    of the best epoch are restored.

    Args:
        model: Calibration head, trained in place.
        train_inputs: Training inputs, first dimension indexing samples.
        train_targets: Training targets aligned with ``train_inputs``.
        val_inputs: Validation inputs used for early stopping.
        val_targets: Validation targets aligned with ``val_inputs``.
        loss_fn: ``(model, inputs, targets) -> (objective, data_loss)``;
            ``objective`` is minimised, ``data_loss`` is the unregularised
            loss monitored on the validation set.
        lr_max: Initial AdamW learning rate.
        lr_min: Cosine scheduler floor learning rate.
        batch_size: Mini-batch size.
        max_epochs: Upper bound on the number of epochs.
        patience: Epochs without validation improvement before stopping.
        shuffle_seed: Seed for the per-epoch batch order; ``None`` leaves it
            unseeded.
        verbose: Show tqdm progress and the loss plot.
        logging: Save the loss plot to ``log_dir``.
        log_dir: Directory for the plot; required when ``logging=True``.
        log_filename: Override the auto-generated plot filename.

    Returns:
        The 1-based epoch whose weights the model ends up with.

    Raises:
        ValueError: If ``logging=True`` and ``log_dir`` is not set.
    """
    if logging and not log_dir:
        raise ValueError("log_dir must be set when logging=True")

    sample_count = train_targets.shape[0]
    steps_per_epoch = math.ceil(sample_count / batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_epochs * steps_per_epoch, lr_min
    )
    generator = torch.Generator()
    if shuffle_seed is not None:
        generator.manual_seed(shuffle_seed)

    best_state, best_val_loss, best_epoch = None, float("inf"), 0
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, max_epochs + 1), desc="Training (epochs)...", disable=not verbose):
        permutation = torch.randperm(sample_count, generator=generator).to(train_targets.device)
        epoch_loss = torch.zeros((), device=train_targets.device)
        for start in range(0, sample_count, batch_size):
            batch_ids = permutation[start : start + batch_size]

            optimizer.zero_grad()
            objective, data_loss = loss_fn(model, train_inputs[batch_ids], train_targets[batch_ids])
            objective.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += data_loss.detach() * batch_ids.shape[0]

        with torch.no_grad():
            _, val_loss = loss_fn(model, val_inputs, val_targets)
        train_losses.append((epoch_loss / sample_count).item())
        val_losses.append(val_loss.item())

        if val_losses[-1] < best_val_loss:
            best_val_loss, best_epoch = val_losses[-1], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)

    if verbose or logging:
        epochs = range(1, len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(epochs, train_losses, label="Train loss", marker="o")
        ax.plot(epochs, val_losses, label="Validation loss", marker="s")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if logging:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fname = log_filename or make_next_indexed_log_filename(
                log_dir=log_dir,
                prefix="calibration_fit_loss",
                extension=".png",
            )
            fig.savefig(os.path.join(log_dir, fname), dpi=200, bbox_inches="tight")
        if verbose:
            plt.show()
        plt.close(fig)

    return best_epoch
