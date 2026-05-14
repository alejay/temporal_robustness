"""
Simple probabilistic state-space baseline with diagonal Gaussian state updates.

The model propagates a latent Gaussian state through time gaps, performs a
Kalman-like diagonal update when observations arrive, and maps the filtered
state to risk with a logistic readout.
"""

import torch
import torch.nn as nn


class StateSpaceRiskModel(nn.Module):
    """Diagonal Gaussian state-space model with filtered risk readout."""

    def __init__(self, input_size: int, state_dim: int = 32):
        super().__init__()
        self.input_size = input_size
        self.state_dim = state_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(input_size * 2, state_dim),
            nn.Tanh(),
            nn.Linear(state_dim, state_dim),
        )
        self.log_decay = nn.Parameter(torch.zeros(state_dim))
        self.log_process_var = nn.Parameter(torch.zeros(state_dim))
        self.log_obs_var = nn.Parameter(torch.zeros(state_dim))
        self.risk_head = nn.Linear(state_dim, 1)

    def _predict_step(self, mean, var, gap):
        """Propagate the latent state across a time gap before assimilation."""
        decay = torch.exp(-torch.nn.functional.softplus(self.log_decay) * gap.unsqueeze(1))
        process_var = torch.nn.functional.softplus(self.log_process_var).unsqueeze(0)
        pred_mean = decay * mean
        pred_var = decay.pow(2) * var + process_var * (gap.unsqueeze(1) + 1.0)
        return pred_mean, pred_var

    def _filter_sequence(self, times, values, mask, obs_mask, lengths):
        """Run diagonal predict-update filtering and collect stepwise diagnostics."""
        B, T, _ = values.shape
        device = values.device
        mean = torch.zeros(B, self.state_dim, device=device)
        var = torch.ones(B, self.state_dim, device=device)

        risk_before_traj = []
        risk_after_traj = []
        delta_h = []
        unc_traj = []

        for t in range(T):
            active = obs_mask[:, t].bool()
            if t == 0:
                gap = torch.zeros(B, device=device)
            else:
                gap = (times[:, t] - times[:, t - 1]).clamp(min=0.0)

            pred_mean, pred_var = self._predict_step(mean, var, gap)
            risk_before = torch.sigmoid(self.risk_head(pred_mean)).squeeze(-1)

            obs_feat = torch.cat([values[:, t, :] * mask[:, t, :], mask[:, t, :]], dim=1)
            obs_mean = self.obs_encoder(obs_feat)
            obs_var = torch.nn.functional.softplus(self.log_obs_var).unsqueeze(0)
            has_obs = (mask[:, t, :].sum(dim=1, keepdim=True) > 0).float()

            post_var = 1.0 / (1.0 / (pred_var + 1e-6) + has_obs / (obs_var + 1e-6))
            post_mean = post_var * (
                pred_mean / (pred_var + 1e-6) + has_obs * obs_mean / (obs_var + 1e-6)
            )
            mean = torch.where(active.unsqueeze(1), post_mean, pred_mean)
            var = torch.where(active.unsqueeze(1), post_var, pred_var)

            risk_after = torch.sigmoid(self.risk_head(mean)).squeeze(-1)
            delta = (risk_after - risk_before).abs() * obs_mask[:, t]
            risk_before_traj.append(risk_before.unsqueeze(1))
            risk_after_traj.append(risk_after.unsqueeze(1))
            delta_h.append(delta.unsqueeze(1))
            unc_traj.append(var.sum(dim=1).unsqueeze(1))

        risk_after_traj = torch.cat(risk_after_traj, dim=1)
        risk_before_traj = torch.cat(risk_before_traj, dim=1)
        delta_h = torch.cat(delta_h, dim=1)
        unc_traj = torch.cat(unc_traj, dim=1)
        idx = (lengths - 1).clamp(min=0)
        gather_idx = idx.view(B, 1)
        final_risk = risk_after_traj.gather(1, gather_idx).squeeze(1)
        final_logit = torch.logit(final_risk.clamp(1e-5, 1 - 1e-5))
        return {
            "logits": final_logit,
            "risk_before": risk_before_traj,
            "risk_after": risk_after_traj,
            "delta_h": delta_h,
            "uncertainty": unc_traj,
        }

    def forward(self, times, values, mask, obs_mask, lengths,
                return_trajectory: bool = False):
        """Return final logits or the full filtered trajectory diagnostics."""
        result = self._filter_sequence(times, values, mask, obs_mask, lengths)
        if return_trajectory:
            return result
        return result["logits"]


def train_state_space(model, train_loader, val_loader, device,
                      n_epochs: int = 30, lr: float = 1e-3,
                      patience: int = 5, verbose: bool = True):
    """Train the state-space model with early stopping on validation AUC."""
    from sklearn.metrics import roc_auc_score

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            times = batch["times"].to(device)
            values = batch["values"].to(device)
            mask = batch["mask"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            logits = model(times, values, mask, obs_mask, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                times = batch["times"].to(device)
                values = batch["values"].to(device)
                mask = batch["mask"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                lengths = batch["lengths"].to(device)
                logits = model(times, values, mask, obs_mask, lengths)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = 0.5

        scheduler.step(1.0 - auc)
        if verbose:
            print(f"  Epoch {epoch+1:3d} | loss {total_loss/len(train_loader):.4f}"
                  f" | val AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
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
