"""
GRU-D: Recurrent Neural Networks for Multivariate Time Series with Missing Values
Che et al. (2018)  https://www.nature.com/articles/s41598-018-24271-9

Key additions for this pilot:
  - MC Dropout for epistemic uncertainty proxy
  - Exposure of per-step hidden states and predicted risk for diagnostic use
  - Evidence of pre/post-update risk change (Δh) at each observation arrival
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GRUD(nn.Module):
    """
    GRU-D with learned temporal decay on both inputs and hidden state.

    Parameters
    ----------
    input_size : int
        Number of features D.
    hidden_size : int
        GRU hidden dimension.
    dropout : float
        MC Dropout rate applied to the hidden state at every step.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.dropout_p   = dropout

        # Learnable log-decay rates for inputs and hidden state (one per feature)
        self.log_gamma_x = nn.Parameter(torch.zeros(input_size))   # input decay
        self.log_gamma_h = nn.Parameter(torch.zeros(hidden_size))  # hidden decay

        # Learnable last-observed-value replacement (empirical mean proxy)
        self.x_mean = nn.Parameter(torch.zeros(input_size))

        # GRU cell operating on [x_tilde; mask; delta_t_norm]
        self.gru_cell = nn.GRUCell(input_size * 2 + 1, hidden_size)

        # Output head
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def _decay(self, delta_t, log_gamma):
        """
        Exponential decay: γ^Δt = exp(−max(0, γ) · Δt).
        delta_t : (..., ) or (..., D) — elapsed time
        log_gamma : (D,) — learnable log decay rates
        """
        gamma = F.softplus(log_gamma)   # positive decay rate
        return torch.exp(-gamma * delta_t)

    # ------------------------------------------------------------------
    def forward(self, times, values, mask, obs_mask, lengths,
                n_mc_samples: int = 1, return_trajectory: bool = False):
        """
        Parameters
        ----------
        times     : (B, T)
        values    : (B, T, D)
        mask      : (B, T, D)   — 1 where observed
        obs_mask  : (B, T)      — 1 where time step is not padding
        lengths   : (B,)
        n_mc_samples : int
            >1 activates MC Dropout inference (multiple stochastic passes).
        return_trajectory : bool
            If True, also return the per-step risk trajectory and hidden states.

        Returns
        -------
        logits : (B,) or (B, n_mc_samples) if n_mc_samples > 1
        If return_trajectory=True additionally returns:
            h_traj   : (B, T, H)   hidden states at each step
            risk_traj: (B, T)      sigmoid(logit) at each step
            delta_h  : (B, T)      |h(t+) - h(t-)| at each observation arrival
        """
        B, T, D = values.shape
        device  = values.device

        def single_pass():
            h = torch.zeros(B, self.hidden_size, device=device)
            last_obs   = self.x_mean.unsqueeze(0).expand(B, -1).clone()
            last_obs_t = torch.zeros(B, D, device=device)

            h_traj    = []
            risk_traj = []
            delta_h   = []

            for t in range(T):
                t_now   = times[:, t]                      # (B,)
                x_t     = values[:, t, :]                  # (B, D)
                m_t     = mask[:, t, :]                    # (B, D)
                active  = obs_mask[:, t].bool()            # (B,)

                # Elapsed time since last observation of each feature
                delta_x = t_now.unsqueeze(1) - last_obs_t  # (B, D)
                delta_x = torch.clamp(delta_x, min=0.0)

                # Elapsed time since last overall step (for hidden decay)
                if t == 0:
                    delta_h_t = torch.zeros(B, 1, device=device)
                else:
                    delta_h_t = (t_now - times[:, t - 1]).unsqueeze(1)
                    delta_h_t = torch.clamp(delta_h_t, min=0.0)

                # Decay inputs and hidden state
                gamma_x = self._decay(delta_x, self.log_gamma_x)        # (B, D)
                gamma_h = self._decay(delta_h_t, self.log_gamma_h)      # (B, H)

                x_tilde = m_t * x_t + (1 - m_t) * (gamma_x * last_obs
                          + (1 - gamma_x) * self.x_mean)
                h_decay = gamma_h * h

                # Hidden state BEFORE update (for Δh diagnostic)
                h_before = h_decay.clone()

                # Apply MC Dropout to hidden state
                h_dropped = self.dropout(h_decay)

                # GRU update: input is [x_tilde, mask, mean_delta_t]
                mean_delta = delta_x.mean(dim=1, keepdim=True)           # (B, 1)
                gru_input  = torch.cat([x_tilde, m_t, mean_delta], dim=1)
                h_new = self.gru_cell(gru_input, h_dropped)

                # Only update active (non-padded) positions
                h = torch.where(active.unsqueeze(1).expand_as(h), h_new, h)

                # Risk at this step (using dropout for MC)
                risk_t = torch.sigmoid(self.fc(self.dropout(h))).squeeze(1)

                if return_trajectory:
                    h_traj.append(h.unsqueeze(1))
                    risk_traj.append(risk_t.unsqueeze(1))
                    # Δh: change in risk due to this observation
                    risk_before = torch.sigmoid(self.fc(h_before)).squeeze(1)
                    delta_h.append((risk_t - risk_before).abs().unsqueeze(1))

                # Update last observed values
                last_obs   = torch.where(m_t.bool(), x_t, last_obs)
                last_obs_t = torch.where(m_t.bool(), t_now.unsqueeze(1)
                                         .expand_as(last_obs_t), last_obs_t)

            # Final prediction from last non-padded hidden state
            idx = (lengths - 1).clamp(min=0)
            if return_trajectory:
                h_final = torch.stack(h_traj, dim=1).squeeze(2)  # (B,T,H)
                # Gather last valid hidden state
                idx_expand = idx.view(B, 1, 1).expand(-1, 1, self.hidden_size)
                h_last = h_final.gather(1, idx_expand).squeeze(1)
            else:
                h_last = h

            logit = self.fc(self.dropout(h_last)).squeeze(1)

            if return_trajectory:
                return (logit,
                        torch.cat(h_traj,    dim=1),
                        torch.cat(risk_traj, dim=1),
                        torch.cat(delta_h,   dim=1))
            return logit

        if n_mc_samples == 1 and not return_trajectory:
            self.train(False)
            with torch.no_grad():
                return single_pass()

        if n_mc_samples > 1:
            # MC Dropout: keep dropout active during inference
            self.train(True)
            samples = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    samples.append(torch.sigmoid(single_pass()).unsqueeze(1))
            return torch.cat(samples, dim=1)   # (B, n_mc_samples)

        # return_trajectory mode
        self.train(True)
        with torch.no_grad():
            return single_pass()


# ── Training utilities ────────────────────────────────────────────────────────

def train_grud(model, train_loader, val_loader, device,
               n_epochs: int = 30, lr: float = 1e-3, pos_weight: float = 4.0,
               patience: int = 5, verbose: bool = True):
    """
    Standard training loop with early stopping on validation AUC-ROC.

    Returns best validation AUC-ROC.
    """
    from sklearn.metrics import roc_auc_score

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            times    = batch["times"].to(device)
            values   = batch["values"].to(device)
            mask     = batch["mask"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            labels   = batch["labels"].to(device)
            lengths  = batch["lengths"].to(device)

            optimizer.zero_grad()
            model.train()
            logits = _forward_train(model, times, values, mask, obs_mask, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                times    = batch["times"].to(device)
                values   = batch["values"].to(device)
                mask     = batch["mask"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                labels   = batch["labels"]
                lengths  = batch["lengths"].to(device)

                logits = _forward_single(model, times, values, mask, obs_mask, lengths)
                preds  = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = 0.5

        scheduler.step(1.0 - auc)

        if verbose:
            print(f"  Epoch {epoch+1:3d} | loss {total_loss/len(train_loader):.4f}"
                  f" | val AUC {auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_auc


def _forward_single(model, times, values, mask, obs_mask, lengths):
    """Single deterministic forward pass (dropout in eval mode)."""
    model.eval()
    with torch.no_grad():
        return model(times, values, mask, obs_mask, lengths, n_mc_samples=1)


def _forward_train(model, times, values, mask, obs_mask, lengths):
    """Single forward pass with dropout active (for training)."""
    # Run forward with dropout active — we call the internal logic directly
    # by setting n_mc_samples=1 but with model in train mode.
    B, T, D = values.shape
    device  = values.device
    h = torch.zeros(B, model.hidden_size, device=device)
    last_obs   = model.x_mean.unsqueeze(0).expand(B, -1).clone()
    last_obs_t = torch.zeros(B, D, device=device)

    for t in range(T):
        t_now   = times[:, t]
        x_t     = values[:, t, :]
        m_t     = mask[:, t, :]
        active  = obs_mask[:, t].bool()

        delta_x = (t_now.unsqueeze(1) - last_obs_t).clamp(min=0.0)
        if t == 0:
            delta_h_t = torch.zeros(B, 1, device=device)
        else:
            delta_h_t = (t_now - times[:, t - 1]).unsqueeze(1).clamp(min=0.0)

        gamma_x = torch.exp(-torch.nn.functional.softplus(model.log_gamma_x) * delta_x)
        gamma_h = torch.exp(-torch.nn.functional.softplus(model.log_gamma_h) * delta_h_t)

        x_tilde = m_t * x_t + (1 - m_t) * (gamma_x * last_obs + (1 - gamma_x) * model.x_mean)
        h_decay = gamma_h * h
        h_drop  = model.dropout(h_decay)

        mean_delta = delta_x.mean(dim=1, keepdim=True)
        gru_input  = torch.cat([x_tilde, m_t, mean_delta], dim=1)
        h_new = model.gru_cell(gru_input, h_drop)
        h = torch.where(active.unsqueeze(1).expand_as(h), h_new, h)

        last_obs   = torch.where(m_t.bool(), x_t, last_obs)
        last_obs_t = torch.where(m_t.bool(), t_now.unsqueeze(1).expand_as(last_obs_t), last_obs_t)

    idx = (lengths - 1).clamp(min=0)
    h_last = h  # final hidden state
    return model.fc(model.dropout(h_last)).squeeze(1)
