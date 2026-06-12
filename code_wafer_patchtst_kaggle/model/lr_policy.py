import torch
from torch import nn


class GRULRPolicy(nn.Module):
    def __init__(self, input_dim, hidden=129, num_layers=3, dropout=0.027):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class ModularLRPolicy(nn.Module):
    def __init__(
        self,
        input_dim=233,
        landscape_dim=30,
        latent_dim=200,
        time_dim=3,
        hidden=64,
        num_layers=2,
        dropout=0.15,
    ):
        super().__init__()
        expected_dim = landscape_dim + latent_dim + time_dim
        if input_dim != expected_dim:
            raise ValueError(f"input_dim={input_dim} but expected {expected_dim}")

        self.landscape_dim = landscape_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.landscape_encoder = nn.Sequential(
            nn.Linear(landscape_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=72,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        landscape = x[..., : self.landscape_dim]
        latent_start = self.landscape_dim
        latent_end = latent_start + self.latent_dim
        latent = x[..., latent_start:latent_end]
        time_features = x[..., latent_end : latent_end + self.time_dim]

        original_shape = landscape.shape[:-1]
        landscape = self.landscape_encoder(landscape.reshape(-1, self.landscape_dim))
        latent = self.latent_encoder(latent.reshape(-1, self.latent_dim))
        time_features = self.time_encoder(time_features.reshape(-1, self.time_dim))
        encoded = torch.cat([landscape, latent, time_features], dim=-1)
        encoded = encoded.reshape(*original_shape, -1)
        out, _ = self.gru(encoded)
        return self.head(out[:, -1, :]).squeeze(-1)


class AttentionModularLRPolicy(nn.Module):
    """Modular per-step encoder followed by a small Transformer encoder over the time window.
    Same input layout as ModularLRPolicy: landscape | latent | time."""

    def __init__(
        self,
        input_dim=233,
        landscape_dim=30,
        latent_dim=200,
        time_dim=3,
        token_dim=72,
        hidden=64,
        nhead=4,
        num_layers=2,
        dropout=0.15,
        max_window=64,
    ):
        super().__init__()
        expected_dim = landscape_dim + latent_dim + time_dim
        if input_dim != expected_dim:
            raise ValueError(f"input_dim={input_dim} but expected {expected_dim}")

        self.landscape_dim = landscape_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.landscape_encoder = nn.Sequential(
            nn.Linear(landscape_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.token_proj = nn.Linear(token_dim, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_window, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        landscape = x[..., : self.landscape_dim]
        latent_start = self.landscape_dim
        latent_end = latent_start + self.latent_dim
        latent = x[..., latent_start:latent_end]
        time_features = x[..., latent_end : latent_end + self.time_dim]

        original_shape = landscape.shape[:-1]
        landscape = self.landscape_encoder(landscape.reshape(-1, self.landscape_dim))
        latent = self.latent_encoder(latent.reshape(-1, self.latent_dim))
        time_features = self.time_encoder(time_features.reshape(-1, self.time_dim))
        tokens = torch.cat([landscape, latent, time_features], dim=-1)
        tokens = tokens.reshape(*original_shape, -1)
        tokens = self.token_proj(tokens)
        seq_len = tokens.shape[1]
        if seq_len > self.position.shape[1]:
            raise ValueError(f"window length {seq_len} exceeds positional capacity {self.position.shape[1]}")
        tokens = tokens + self.position[:, :seq_len]
        encoded = self.transformer(tokens)
        return self.head(encoded[:, -1, :]).squeeze(-1)


class TemporalBlock(nn.Module):
    """Causal dilated 1D-conv block with residual connection (TCN building block)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)[..., :-self.pad] if self.pad > 0 else self.conv1(x)
        out = self.dropout(self.act(out))
        out = self.conv2(out)[..., :-self.pad] if self.pad > 0 else self.conv2(out)
        out = self.dropout(self.act(out))
        return out + self.residual(x)


class TCNLRPolicy(nn.Module):
    """Temporal Convolutional Network with dilated causal 1D convolutions.

    Reference: Bai, Kolter, Koltun (2018) "An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling".  Receptive
    field grows exponentially with depth via dilation; often beats LSTM/GRU
    on short-to-medium sequences with similar parameter budgets.
    """

    def __init__(self, input_dim=230, hidden=64, num_layers=4, kernel_size=3, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([
            TemporalBlock(hidden, hidden, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.head(h.transpose(1, 2)[:, -1, :]).squeeze(-1)


class PatchTSTLRPolicy(nn.Module):
    """Channel-independent transformer over patches of the input window.

    Reference: Nie et al. NeurIPS 2023 "A Time Series is Worth 64 Words:
    Long-term Forecasting with Transformers" (PatchTST).  We flatten by
    treating each of the ``input_dim`` features as an independent channel,
    split its time-window into patches, run a small Transformer per channel,
    take the last patch representation, then aggregate channel scalars
    through a final linear head.
    """

    def __init__(self, input_dim=230, patch_size=2, hidden=16, nhead=4,
                 num_layers=2, dropout=0.15, max_window=20):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        max_patches = (max_window + patch_size - 1) // patch_size
        self.patch_embed = nn.Linear(patch_size, hidden)
        self.position = nn.Parameter(torch.zeros(1, max_patches, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.channel_head = nn.Linear(hidden, 1)
        self.final = nn.Linear(input_dim, 1)

    def forward(self, x):
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"PatchTST got D={D}, expected {self.input_dim}")
        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad:
            x = nn.functional.pad(x, (0, 0, 0, pad))
        T_padded = T + pad
        n_patches = T_padded // self.patch_size
        x = x.transpose(1, 2).reshape(B * D, n_patches, self.patch_size)
        x = self.patch_embed(x) + self.position[:, :n_patches]
        x = self.transformer(x)[:, -1, :]
        x = self.channel_head(x).squeeze(-1).reshape(B, D)
        return self.final(x).squeeze(-1)


class NBeatsBlock(nn.Module):
    """One generic N-BEATS block: shared 4-layer FC trunk → backcast + forecast heads."""

    def __init__(self, flat_dim, hidden, dropout):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.theta_back = nn.Linear(hidden, flat_dim)
        self.theta_fore = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.theta_back(h), self.theta_fore(h).squeeze(-1)


class NBeatsLRPolicy(nn.Module):
    """N-BEATS-style stacked residual fully-connected blocks.

    Reference: Oreshkin et al. ICLR 2020 "N-BEATS: Neural basis expansion
    analysis for interpretable time series forecasting".  Won M4 competition
    by a large margin against statistical & deep baselines.  We use the
    generic (non-interpretable) variant: each block predicts a backcast
    (subtracted from the input residual) and a forecast contribution; final
    prediction is the sum of all block forecasts.
    """

    def __init__(self, input_dim=230, window=10, hidden=128, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.window = window
        self.flat_dim = input_dim * window
        self.blocks = nn.ModuleList([
            NBeatsBlock(self.flat_dim, hidden, dropout) for _ in range(num_blocks)
        ])

    def forward(self, x):
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"NBeats expected input_dim={self.input_dim}, got {D}")
        if T < self.window:
            pad = x[:, :1, :].expand(B, self.window - T, D)
            x = torch.cat([pad, x], dim=1)
        elif T > self.window:
            x = x[:, -self.window:, :]
        residual = x.reshape(B, -1)
        forecast = 0.0
        for block in self.blocks:
            back, fore = block(residual)
            residual = residual - back
            forecast = forecast + fore
        return forecast


class DLinearLRPolicy(nn.Module):
    """DLinear: series decomposition (trend = moving avg) + 2 parallel linear heads.

    Reference: Zeng et al. AAAI 2023 "Are Transformers Effective for Time Series
    Forecasting?" — surprising minimal baseline that beats most transformer
    variants on long-horizon forecasting benchmarks.  Cheapest learnable LR
    policy possible while still using a non-trivial inductive bias.
    """

    def __init__(self, input_dim=230, window=10, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.window = window
        pad = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size, stride=1, padding=pad, count_include_pad=False)
        self.linear_trend = nn.Linear(window, 1)
        self.linear_seasonal = nn.Linear(window, 1)
        self.channel_proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"DLinear expected input_dim={self.input_dim}, got {D}")
        if T < self.window:
            pad = x[:, :1, :].expand(B, self.window - T, D)
            x = torch.cat([pad, x], dim=1)
        elif T > self.window:
            x = x[:, -self.window:, :]
        x_perm = x.transpose(1, 2)
        trend = self.avg_pool(x_perm)
        seasonal = x_perm - trend
        trend_pred = self.linear_trend(trend).squeeze(-1)
        seasonal_pred = self.linear_seasonal(seasonal).squeeze(-1)
        return self.channel_proj(trend_pred + seasonal_pred).squeeze(-1)


class CurveLRPolicy(nn.Module):
    """MPC-style policy that predicts the loss-landscape curve over LR_GRID for the
    current and next ``lookahead_n - 1`` epochs.

    Input layout matches ModularLRPolicy (landscape | latent | time).
    Output shape is (B, lookahead_n, landscape_dim) where landscape_dim equals the
    number of LR_GRID points (30 in the current oracle).
    """

    def __init__(
        self,
        input_dim=233,
        landscape_dim=30,
        latent_dim=200,
        time_dim=3,
        hidden=128,
        num_layers=2,
        dropout=0.15,
        lookahead_n=2,
    ):
        super().__init__()
        expected_dim = landscape_dim + latent_dim + time_dim
        if input_dim != expected_dim:
            raise ValueError(f"input_dim={input_dim} but expected {expected_dim}")
        if lookahead_n < 1:
            raise ValueError("lookahead_n must be >= 1")

        self.landscape_dim = landscape_dim
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.lookahead_n = lookahead_n

        self.landscape_encoder = nn.Sequential(
            nn.Linear(landscape_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=72,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, lookahead_n * landscape_dim),
        )

    def forward(self, x):
        landscape = x[..., : self.landscape_dim]
        latent_start = self.landscape_dim
        latent_end = latent_start + self.latent_dim
        latent = x[..., latent_start:latent_end]
        time_features = x[..., latent_end : latent_end + self.time_dim]

        original_shape = landscape.shape[:-1]
        landscape = self.landscape_encoder(landscape.reshape(-1, self.landscape_dim))
        latent = self.latent_encoder(latent.reshape(-1, self.latent_dim))
        time_features = self.time_encoder(time_features.reshape(-1, self.time_dim))
        encoded = torch.cat([landscape, latent, time_features], dim=-1)
        encoded = encoded.reshape(*original_shape, -1)
        out, _ = self.gru(encoded)
        last = out[:, -1, :]
        head = self.head(last)
        head = head.view(head.shape[0], self.lookahead_n, self.landscape_dim)
        return head
