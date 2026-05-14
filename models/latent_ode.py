"""
Latent ODE with ODE-RNN encoder for irregular time series.
Based on: Rubanova et al. (2019) "Latent ODEs for Irregularly-Sampled Time Series"
https://arxiv.org/abs/1907.03907

Architecture:
  Encoder  : ODE-RNN (GRU + Neural ODE) running backwards through observations
  Latent   : Gaussian q(z0 | x) → sample z0
  Decoder  : Neural ODE evolves z(t) forward; linear readout → risk trajectory
  Training : ELBO = reconstruction log-likelihood − KL(q || p)

Key additions for this pilot:
  - Full risk trajectory h(t) accessible at arbitrary query times
  - Epistemic uncertainty from the posterior variance over z0
  - Δh diagnostic: difference in predicted risk pre/post observation incorporation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np


# ── ODE dynamics ──────────────────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """Small MLP defining dz/dt = f(z, t)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Initialise to near-zero dynamics for training stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)


# ── Encoder: ODE-RNN ──────────────────────────────────────────────────────────

class ODERNNEncoder(nn.Module):
    """
    Encodes an irregular time series backwards using a GRU cell interleaved
    with a Neural ODE to propagate the hidden state between observation times.
    Outputs parameters of q(z0): (mu, log_var) each of shape (B, latent_dim).
    """

    def __init__(self, input_size: int, hidden_dim: int = 64,
                 latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.ode_func   = ODEFunc(hidden_dim, hidden_dim)
        self.gru_cell   = nn.GRUCell(input_size * 2, hidden_dim)   # [x; mask]
        self.to_mu      = nn.Linear(hidden_dim, latent_dim)
        self.to_log_var = nn.Linear(hidden_dim, latent_dim)

    def _ode_step(self, h, t_start, t_end):
        """Evolve h from t_start to t_end using the ODE."""
        if (t_end - t_start).abs() < 1e-4:
            return h
        t_span = torch.stack([t_start, t_end])
        # odeint expects (T, B, D) output
        h_out = odeint(self.ode_func, h, t_span,
                       method="euler", options={"step_size": 0.1})
        return h_out[-1]

    def forward(self, times, values, mask, obs_mask):
        """
        times    : (B, T)
        values   : (B, T, D)
        mask     : (B, T, D)
        obs_mask : (B, T)

        Process observations in reverse chronological order.
        """
        B, T, D = values.shape
        device  = values.device
        h = torch.zeros(B, self.hidden_dim, device=device)

        for t in reversed(range(T)):
            active = obs_mask[:, t].bool()
            t_now  = times[:, t]

            if t < T - 1:
                t_next = times[:, t + 1]
                h = self._ode_step(h,
                                   t_next.mean(),   # scalar approximation
                                   t_now.mean())

            # Update with GRU at this observation
            x_t    = values[:, t, :]
            m_t    = mask[:, t, :]
            inp    = torch.cat([x_t * m_t, m_t], dim=1)   # zero out missing
            h_new  = self.gru_cell(inp, h)
            h = torch.where(active.unsqueeze(1).expand_as(h), h_new, h)

        mu      = self.to_mu(h)
        log_var = self.to_log_var(h).clamp(-8, 4)
        return mu, log_var


# ── Decoder: ODE forward evolution ───────────────────────────────────────────

class ODEDecoder(nn.Module):
    """
    Given z0 ~ q(z0), evolve z(t) forward with a Neural ODE and map to risk.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        self.readout  = nn.Linear(latent_dim, 1)

    def forward(self, z0, query_times):
        """
        z0          : (B, latent_dim)
        query_times : (Q,) sorted query times (shared across batch)

        Returns
        -------
        z_traj   : (B, Q, latent_dim)
        risk_traj: (B, Q)  sigmoid of linear readout
        """
        z_traj = odeint(self.ode_func, z0, query_times,
                        method="euler", options={"step_size": 0.1})
        # z_traj : (Q, B, latent_dim) → (B, Q, latent_dim)
        z_traj = z_traj.permute(1, 0, 2)
        risk   = torch.sigmoid(self.readout(z_traj)).squeeze(-1)  # (B, Q)
        return z_traj, risk


# ── Full Latent ODE model ─────────────────────────────────────────────────────

class LatentODE(nn.Module):
    """
    Full Latent ODE model for binary outcome prediction.

    Exposes:
      - risk_at_times()  : predicted risk at arbitrary query times
      - uncertainty()    : marginal epistemic uncertainty via posterior variance
      - delta_h()        : update in predicted risk when new obs are incorporated
    """

    def __init__(self, input_size: int, hidden_dim: int = 64,
                 latent_dim: int = 32, n_samples: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples  = n_samples

        self.encoder = ODERNNEncoder(input_size, hidden_dim, latent_dim)
        self.decoder = ODEDecoder(latent_dim, hidden_dim)

    # ── reparameterisation ────────────────────────────────────────────────────

    def _reparam(self, mu, log_var):
        """Sample z0 from the approximate posterior via reparameterization."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── ELBO ─────────────────────────────────────────────────────────────────

    def elbo(self, times, values, mask, obs_mask, labels, lengths):
        """
        Compute ELBO = E_q[log p(y | z0)] − KL(q || p).
        Uses the final predicted risk as the outcome probability.
        """
        B = times.shape[0]
        device = times.device

        mu, log_var = self.encoder(times, values, mask, obs_mask)
        z0 = self._reparam(mu, log_var)

        # Query at the last valid time for each patient
        # Use a single common time grid spanning [0, max_time]
        max_t   = times.max().item()
        t_grid  = torch.linspace(0, max_t, steps=20, device=device)
        _, risk = self.decoder(z0, t_grid)          # (B, 20)

        # Use risk at the last observation time
        idx = (lengths - 1).clamp(min=0)
        # Map length index to t_grid index (approximate)
        t_grid_idx = ((times[torch.arange(B), idx] / max_t)
                      * (t_grid.shape[0] - 1)).long().clamp(0, t_grid.shape[0]-1)
        risk_final = risk[torch.arange(B), t_grid_idx]

        # Reconstruction log-likelihood
        ll = (labels * torch.log(risk_final.clamp(1e-7, 1 - 1e-7))
              + (1 - labels) * torch.log((1 - risk_final).clamp(1e-7, 1 - 1e-7)))

        # KL divergence (analytical for Gaussian)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)

        return -(ll - 0.01 * kl).mean()

    # ── Prediction ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, times, values, mask, obs_mask, lengths,
                n_samples: int = None):
        """
        Returns mean predicted risk (B,) and epistemic uncertainty (B,).
        Uncertainty = variance of risk over posterior samples.
        """
        n_samples = n_samples or self.n_samples
        self.eval()

        mu, log_var = self.encoder(times, values, mask, obs_mask)
        max_t  = times.max().item()
        t_grid = torch.linspace(0, max_t, steps=20, device=times.device)

        B      = times.shape[0]
        risks  = []
        for _ in range(n_samples):
            z0 = self._reparam(mu, log_var)
            _, risk = self.decoder(z0, t_grid)     # (B, 20)
            idx = (lengths - 1).clamp(min=0)
            t_grid_idx = ((times[torch.arange(B), idx] / max(max_t, 1e-4))
                          * (t_grid.shape[0] - 1)).long().clamp(0, t_grid.shape[0]-1)
            risks.append(risk[torch.arange(B), t_grid_idx])

        risks_stack = torch.stack(risks, dim=1)   # (B, n_samples)
        mean_risk   = risks_stack.mean(dim=1)
        epistemic_u = risks_stack.var(dim=1)
        return mean_risk, epistemic_u

    @torch.no_grad()
    def risk_trajectory(self, times, values, mask, obs_mask, lengths,
                        query_times: torch.Tensor = None, n_samples: int = 1):
        """
        Return predicted risk trajectory h(t) for a batch.

        query_times : (Q,) — if None, uses a uniform grid over [0, max_t]

        Returns
        -------
        risk_mean : (B, Q)   mean risk at each query time
        risk_std  : (B, Q)   std of risk at each query time (epistemic)
        query_t   : (Q,)     the query times used
        """
        self.eval()
        mu, log_var = self.encoder(times, values, mask, obs_mask)

        max_t = times.max().item()
        if query_times is None:
            query_times = torch.linspace(0, max_t, steps=50, device=times.device)

        all_risks = []
        for _ in range(n_samples):
            z0 = self._reparam(mu, log_var)
            _, risk = self.decoder(z0, query_times)   # (B, Q)
            all_risks.append(risk)

        risks_stack = torch.stack(all_risks, dim=0)   # (n_samples, B, Q)
        return risks_stack.mean(0), risks_stack.std(0), query_times

    @torch.no_grad()
    def compute_delta_h(self, times, values, mask, obs_mask, lengths):
        """
        Compute |Δh_i| = |h(t_i+) − h(t_i-)| at each observation time t_i
        by comparing the risk trajectory with and without each observation.

        Returns
        -------
        delta_h  : (B, T)  absolute change in predicted risk at each step
        gap_lens : (B, T)  observation gap lengths preceding each step
        u_before : (B, T)  epistemic uncertainty proxy before each update
        """
        self.eval()
        B, T, D = values.shape
        device  = times.device

        delta_h_list  = []
        gap_list      = []
        u_before_list = []

        for t in range(T):
            active = obs_mask[:, t].bool()

            # h(t_i-): encode using observations up to t-1
            if t == 0:
                risk_before = torch.zeros(B, device=device)
                u_before    = torch.zeros(B, device=device)
                gap         = torch.zeros(B, device=device)
            else:
                mask_before  = mask.clone()
                mask_before[:, t:, :] = 0.0
                obs_before   = obs_mask.clone()
                obs_before[:, t:] = 0.0
                len_before   = (obs_before.sum(dim=1).long()).clamp(min=1)

                mu_b, lv_b = self.encoder(times, values, mask_before, obs_before)
                max_t = times[:, :t].max().item() + 1e-4
                t_query = torch.tensor([max_t], device=device)
                z0_b = mu_b  # use mean for stability
                _, r_b = self.decoder(z0_b, t_query)
                risk_before = r_b[:, 0]

                # Epistemic uncertainty = sum of posterior variances
                u_before = lv_b.exp().sum(dim=1)

                gap = (times[:, t] - times[:, t - 1]).clamp(min=0)

            # h(t_i+): encode including observation t
            mask_after  = mask.clone()
            mask_after[:, t+1:, :] = 0.0
            obs_after   = obs_mask.clone()
            obs_after[:, t+1:] = 0.0

            mu_a, _ = self.encoder(times, values, mask_after, obs_after)
            max_t_a = times[:, :t+1].max().item() + 1e-4
            t_query_a = torch.tensor([max_t_a], device=device)
            _, r_a = self.decoder(mu_a, t_query_a)
            risk_after = r_a[:, 0]

            dh = (risk_after - risk_before).abs()
            dh = torch.where(active, dh, torch.zeros_like(dh))

            delta_h_list.append(dh.unsqueeze(1))
            gap_list.append(gap.unsqueeze(1))
            u_before_list.append(u_before.unsqueeze(1))

        return (torch.cat(delta_h_list, dim=1),
                torch.cat(gap_list,     dim=1),
                torch.cat(u_before_list, dim=1))


# ── Training ──────────────────────────────────────────────────────────────────

def train_latent_ode(model, train_loader, val_loader, device,
                     n_epochs: int = 30, lr: float = 1e-3,
                     patience: int = 5, verbose: bool = True):
    """Train the latent ODE with ELBO optimization and AUC-based stopping."""
    from sklearn.metrics import roc_auc_score

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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
            mask_b   = batch["mask"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            labels   = batch["labels"].to(device)
            lengths  = batch["lengths"].to(device)

            optimizer.zero_grad()
            loss = model.elbo(times, values, mask_b, obs_mask, labels, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                times    = batch["times"].to(device)
                values   = batch["values"].to(device)
                mask_b   = batch["mask"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                lengths  = batch["lengths"].to(device)

                mean_risk, _ = model.predict(
                    times, values, mask_b, obs_mask, lengths)
                all_preds.extend(mean_risk.cpu().numpy())
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
