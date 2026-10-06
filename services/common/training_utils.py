import copy
import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state, vmap
import torch.nn.functional as F
from tqdm import tqdm

from .logging_utils import make_next_indexed_log_filename

# torch.optim.AdamW defaults, reproduced by ``fit_stacked_with_early_stopping``.
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 1e-2


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
    max_epochs: int = 50,
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


def fit_stacked_with_early_stopping(
    models: Sequence[nn.Module],
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    lr_max: Sequence[float],
    lr_min: Sequence[float],
    l1_lambda: Sequence[float],
    l2_lambda: Sequence[float],
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 3,
    shuffle_seed: Optional[int] = None,
) -> List[int]:
    """
    Trains several calibration heads of one shape jointly, each as
    ``fit_with_early_stopping`` would train it alone.

    The heads' parameters are stacked and every step updates all of them at
    once: binary cross-entropy plus each head's L1/L2 penalty, AdamW with
    torch's defaults and a per-head cosine schedule (``CosineAnnealingLR``'s
    recursive form). All heads see the same per-epoch batch order, drawn
    from ``shuffle_seed`` as in ``fit_with_early_stopping``. Early stopping
    is tracked per head on the unregularised validation BCE; a stopped head
    is frozen while the others continue. On CUDA one step is captured into a
    CUDA graph and replayed, so a step costs a single launch instead of a
    launch per tiny kernel. Results match sequential training up to
    floating-point rounding.

    Args:
        models: Calibration heads of the same class and shape, trained in place.
        train_inputs: Training inputs, first dimension indexing samples.
        train_targets: Float training targets aligned with ``train_inputs``.
        val_inputs: Validation inputs used for early stopping.
        val_targets: Float validation targets aligned with ``val_inputs``.
        lr_max: Initial AdamW learning rate per head.
        lr_min: Cosine scheduler floor learning rate per head.
        l1_lambda: L1 penalty strength per head (0 disables).
        l2_lambda: L2 penalty strength per head (0 disables).
        batch_size: Mini-batch size shared by all heads.
        max_epochs: Upper bound on the number of epochs.
        patience: Epochs without validation improvement before a head stops.
        shuffle_seed: Seed for the per-epoch batch order; ``None`` leaves it
            unseeded.

    Returns:
        The 1-based epoch whose weights each head ends up with.
    """
    device = train_inputs.device
    dtype = train_inputs.dtype
    models_count = len(models)
    params, buffers = stack_module_state(list(models)) # name -> [M, ...]
    names = list(params)
    meta_model = copy.deepcopy(models[0]).to("meta")
    forward = vmap(
        lambda p, b, x: functional_call(meta_model, (p, b), (x,)),
        in_dims=(0, 0, None),
    ) # [M, B]

    sample_count = train_targets.shape[0]
    steps_per_epoch = math.ceil(sample_count / batch_size)
    total_steps = max_epochs * steps_per_epoch
    lr = np.array(lr_max, dtype=np.float64)
    lr_floor = np.array(lr_min, dtype=np.float64)
    l1 = torch.tensor(l1_lambda, device=device, dtype=dtype) # [M]
    l2 = torch.tensor(l2_lambda, device=device, dtype=dtype) # [M]
    exp_avg = {k: torch.zeros_like(v) for k, v in params.items()}
    exp_avg_sq = {k: torch.zeros_like(v) for k, v in params.items()}

    generator = torch.Generator()
    if shuffle_seed is not None:
        generator.manual_seed(shuffle_seed)

    def per_head(values, name):
        """Broadcasts a ``[M]`` vector over the stacked parameter ``name``."""
        return values.view(-1, *([1] * (params[name].dim() - 1)))

    def train_step(batch_ids, step_lr, bias_correction1, bias_correction2_sqrt):
        pred = forward(params, buffers, train_inputs[batch_ids]) # [M, B]
        targets = train_targets[batch_ids].expand_as(pred)
        objective = F.binary_cross_entropy(pred, targets, reduction="none").mean(1) # [M]
        if max(l1_lambda) > 0:
            objective = objective + l1 * sum(params[k].abs().reshape(models_count, -1).sum(1) for k in names)
        if max(l2_lambda) > 0:
            objective = objective + l2 * sum((params[k] ** 2).reshape(models_count, -1).sum(1) for k in names)
        # Heads share no parameters, so the gradient of the sum is each head's own gradient.
        grads = torch.autograd.grad(objective.sum(), [params[k] for k in names])
        with torch.no_grad():
            for k, grad in zip(names, grads):
                params[k].mul_(1 - per_head(step_lr, k) * ADAMW_WEIGHT_DECAY)
                exp_avg[k].lerp_(grad, 1 - ADAMW_BETAS[0])
                exp_avg_sq[k].mul_(ADAMW_BETAS[1]).addcmul_(grad, grad, value=1 - ADAMW_BETAS[1])
                denom = (exp_avg_sq[k].sqrt() / bias_correction2_sqrt).add_(ADAMW_EPS)
                params[k].sub_(per_head(step_lr / bias_correction1, k) * exp_avg[k] / denom)

    graph = None
    if device.type == "cuda":
        static_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        static_lr = torch.zeros(models_count, dtype=dtype, device=device)
        static_bc1 = torch.ones((), dtype=dtype, device=device)
        static_bc2 = torch.ones((), dtype=dtype, device=device)
        state = [*params.values(), *exp_avg.values(), *exp_avg_sq.values()]
        initial_state = [t.detach().clone() for t in state]
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3): # warm-up required before capture; rolled back below
                train_step(static_ids, static_lr, static_bc1, static_bc2)
        torch.cuda.current_stream().wait_stream(side_stream)
        with torch.no_grad():
            for tensor, initial in zip(state, initial_state):
                tensor.copy_(initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            train_step(static_ids, static_lr, static_bc1, static_bc2)

    best_state = {k: v.detach().clone() for k, v in params.items()}
    best_val_loss = np.full(models_count, np.inf)
    best_epoch = np.zeros(models_count, dtype=int)
    epochs_without_improvement = np.zeros(models_count, dtype=int)
    active = np.ones(models_count, dtype=bool)
    step = 0

    for epoch in range(1, max_epochs + 1):
        permutation = torch.randperm(sample_count, generator=generator).to(device)
        # Learning rates and bias corrections of the whole epoch are sent in
        # one transfer, so the step loop never waits on the host.
        lr_rows, bias_corrections = [], []
        for _ in range(steps_per_epoch):
            step += 1
            lr_rows.append(np.where(active, lr, 0.0)) # stopped heads are frozen
            bias_corrections.append((1 - ADAMW_BETAS[0] ** step, math.sqrt(1 - ADAMW_BETAS[1] ** step)))
            lr = (1 + math.cos(math.pi * step / total_steps)) \
                / (1 + math.cos(math.pi * (step - 1) / total_steps)) * (lr - lr_floor) + lr_floor
        lr_table = torch.tensor(np.array(lr_rows), dtype=dtype, device=device) # [S, M]
        bias_table = torch.tensor(bias_corrections, dtype=dtype, device=device) # [S, 2]

        for batch_idx, start in enumerate(range(0, sample_count, batch_size)):
            batch_ids = permutation[start : start + batch_size]
            if graph is not None and batch_ids.shape[0] == batch_size:
                static_ids.copy_(batch_ids)
                static_lr.copy_(lr_table[batch_idx])
                static_bc1.copy_(bias_table[batch_idx, 0])
                static_bc2.copy_(bias_table[batch_idx, 1])
                graph.replay()
            else:
                train_step(batch_ids, lr_table[batch_idx], *bias_corrections[batch_idx])

        with torch.no_grad():
            pred = forward(params, buffers, val_inputs) # [M, N_val]
            val_losses = F.binary_cross_entropy(
                pred, val_targets.expand_as(pred), reduction="none"
            ).mean(1).cpu().numpy() # [M]
        for head in np.flatnonzero(active):
            if val_losses[head] < best_val_loss[head]:
                best_val_loss[head], best_epoch[head] = val_losses[head], epoch
                epochs_without_improvement[head] = 0
                for k in names:
                    best_state[k][head] = params[k][head].detach()
            else:
                epochs_without_improvement[head] += 1
                if epochs_without_improvement[head] >= patience:
                    active[head] = False
        if not active.any():
            break

    for head, model in enumerate(models):
        # Buffers (e.g. input standardization) are not trained and stay as they are.
        state = model.state_dict()
        state.update({k: best_state[k][head].clone() for k in names})
        model.load_state_dict(state)
    return best_epoch.tolist()
