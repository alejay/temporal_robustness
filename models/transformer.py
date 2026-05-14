"""
Time-aware transformer baseline for irregular multivariate time series.

This model treats each observation time step as a token whose representation
includes values, missingness mask, absolute time, and local gap length.
MC Dropout over the encoder/output head is used as an epistemic uncertainty
proxy.
"""

import torch
import torch.nn as nn


class TimeAwareTransformer(nn.Module):
    """Time-aware transformer with dropout-based uncertainty estimates."""

    def __init__(self, input_size: int, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.token_proj = nn.Linear(input_size * 2 + 2, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def _build_tokens(self, times, values, mask):
        """Embed each time step from values, masks, absolute time, and gap."""
        B, T, _ = values.shape
        if T == 0:
            raise ValueError("empty sequence")
        gaps = torch.zeros(B, T, device=values.device, dtype=values.dtype)
        if T > 1:
            gaps[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp(min=0.0)
        time_feats = torch.stack([times, gaps], dim=-1)
        token_input = torch.cat([values, mask, time_feats], dim=-1)
        tokens = self.token_proj(token_input) + self.time_mlp(time_feats)
        return self.norm(tokens), gaps

    def _single_pass(self, times, values, mask, obs_mask, lengths,
                     return_trajectory: bool = False):
        """Run one encoder pass and optionally return stepwise diagnostics."""
        tokens, _ = self._build_tokens(times, values, mask)
        src_key_padding_mask = obs_mask < 0.5
        encoded = self.encoder(self.dropout(tokens),
                               src_key_padding_mask=src_key_padding_mask)
        encoded = self.norm(encoded)
        risk_traj = torch.sigmoid(self.fc(self.dropout(encoded))).squeeze(-1)

        last_idx = (lengths - 1).clamp(min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        last_hidden = encoded.gather(1, gather_idx).squeeze(1)
        logits = self.fc(self.dropout(last_hidden)).squeeze(-1)

        if not return_trajectory:
            return logits

        risk_before = torch.cat(
            [torch.zeros(risk_traj.shape[0], 1, device=risk_traj.device),
             risk_traj[:, :-1]],
            dim=1,
        )
        delta_h = (risk_traj - risk_before).abs() * obs_mask
        return logits, encoded, risk_traj, delta_h

    def forward(self, times, values, mask, obs_mask, lengths,
                n_mc_samples: int = 1, return_trajectory: bool = False):
        """Return logits, MC-dropout samples, or diagnostic trajectories."""
        if n_mc_samples == 1 and not return_trajectory:
            self.eval()
            with torch.no_grad():
                return self._single_pass(times, values, mask, obs_mask, lengths)

        if n_mc_samples > 1:
            self.train(True)
            samples = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    samples.append(
                        torch.sigmoid(
                            self._single_pass(times, values, mask, obs_mask, lengths)
                        ).unsqueeze(1)
                    )
            return torch.cat(samples, dim=1)

        self.train(True)
        with torch.no_grad():
            return self._single_pass(
                times, values, mask, obs_mask, lengths, return_trajectory=True
            )


def _forward_single(model, times, values, mask, obs_mask, lengths):
    """Single deterministic forward pass used during evaluation."""
    model.eval()
    with torch.no_grad():
        return model(times, values, mask, obs_mask, lengths, n_mc_samples=1)


def _forward_train(model, times, values, mask, obs_mask, lengths):
    """Single training-time pass with dropout active."""
    model.train()
    return model._single_pass(times, values, mask, obs_mask, lengths)


def train_transformer(model, train_loader, val_loader, device,
                      n_epochs: int = 30, lr: float = 1e-3,
                      patience: int = 5, verbose: bool = True):
    """Train the transformer with early stopping on validation AUC."""
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
            logits = _forward_train(model, times, values, mask, obs_mask, lengths)
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
                logits = _forward_single(model, times, values, mask, obs_mask, lengths)
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
