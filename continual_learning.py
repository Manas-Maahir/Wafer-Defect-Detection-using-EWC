import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Usage:
        ewc = EWC(model, dataloader, device=DEVICE)
        ewc.register_prior_task()          # call AFTER training task N
        loss = selective_ewc_loss(model, task_loss, ewc, ewc_lambda=500)
    """

    def __init__(self, model: nn.Module, dataloader, device="cpu"):
        self.model      = model
        self.dataloader = dataloader
        self.device     = device

        # Snapshot of trainable parameter names → values (θ*)
        self._means: dict[str, torch.Tensor] = {}
        # Fisher diagonal (F_i)
        self._precision_matrices: dict[str, torch.Tensor] = {}

        # Capture current parameter values as the "anchor" means
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone().detach().to(self.device)

    # ── private ──────────────────────────────────────────────
    def _diag_fisher(self) -> dict[str, torch.Tensor]:
        """Estimates the diagonal of the Fisher Information Matrix."""
        precision_matrices: dict[str, torch.Tensor] = {}

        # Initialise accumulators with zeros in the right shape
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                precision_matrices[n] = torch.zeros_like(p.data, device=self.device)

        self.model.eval()
        num_batches = len(self.dataloader)

        for inputs, labels in self.dataloader:
            self.model.zero_grad()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            output = self.model(inputs)
            loss   = F.cross_entropy(output, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    precision_matrices[n] += p.grad.data ** 2

        # Normalise by number of batches
        for n in precision_matrices:
            precision_matrices[n] /= max(num_batches, 1)

        return precision_matrices

    # ── public ───────────────────────────────────────────────
    def register_prior_task(self):
        """
        Call this AFTER training on a task to lock in the Fisher Information
        and the current parameter means (θ*) for EWC regularisation going forward.
        """
        # Update means to current weights
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone().detach().to(self.device)

        # Compute Fisher diagonals
        self._precision_matrices = self._diag_fisher()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Returns the EWC penalty:  Σ_i  F_i · (θ_i - θ*_i)²
        (lambda is applied by the caller via selective_ewc_loss)
        """
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._precision_matrices:
                delta = p - self._means[n]
                loss  = loss + (self._precision_matrices[n] * delta ** 2).sum()
        return loss


# ─────────────────────────────────────────────────────────────
def selective_ewc_loss(
    model:      nn.Module,
    task_loss:  torch.Tensor,
    ewc_object: "EWC | None",
    ewc_lambda: float = 500,
) -> torch.Tensor:
    """
    Adds the EWC penalty to the current task loss.

    Args:
        model:       The model being trained.
        task_loss:   Cross-entropy (or other) loss for the current task.
        ewc_object:  An EWC instance (or None to disable).
        ewc_lambda:  Regularisation strength.  Typical range: 100 – 5000.

    Returns:
        Scalar tensor:  task_loss + ewc_lambda * EWC_penalty
    """
    if ewc_object is None or not ewc_object._precision_matrices:
        return task_loss

    ewc_penalty = ewc_object.penalty(model)
    return task_loss + ewc_lambda * ewc_penalty
