#!/usr/bin/env python3
"""
(1) Train a supervised spectral regressor f(x) that maps a spectrum x -> point estimate \hat{theta}.
(2) Use Simulation-Based Inference (SBI, SNPE + MAF) to learn a calibrated posterior:
        p(theta | \hat{theta})

This is useful when:
- You want FAST point estimates from a neural network (stage 1),
- But you want a FULL posterior (uncertainty, correlations) (stage 2),
- And you can generate training simulations (theta, spectrum) pairs.

----------------------------------------------------------------------
Expected simulation HDF5 format:
- spec   : (N, L)  simulated spectra (float)
- wave   : (L,)    wavelength/velocity grid (float)
- params : (N, P)  ground-truth parameters (float)

Optional observation TXT format:
- columns: wave, flux, err  (Nobs x 3)

Outputs (written to --outdir):
- regressor.pt             (torch weights)
- snpe_estimator.pt        (torch weights)
- theta_hat_obs.npy        (regressor prediction on observation)
- posterior_samples_obs.npy
- regressor_losses.png
- posterior_corner.png
- rank_hist.png
- ranks.npy

----------------------------------------------------------------------
Regressor options:
- --regressor cnn     : built-in 1D CNN regressor (no external libs beyond torch)
- --regressor spender : uses spender.SpectrumEncoder (if installed), with an MLP head

Note: For spender, you must have `spender` installed and compatible with your environment.

----------------------------------------------------------------------
Usage example:

python ppd_sbi_pipeline.py \
  --sim_h5 data/simulations.h5 \
  --obs_txt data/observation.txt \
  --outdir outputs/run1 \
  --regressor cnn \
  --prior_low 0 0.06 25 -1.8 32 32 27.5 28 0 \
  --prior_high 6 0.3 75 0.0 36 36 32.5 32 90 \
  --log10_params 3 \
  --device cuda

If you have no observation:
python ppd_sbi_pipeline.py --sim_h5 data/simulations.h5 --outdir outputs/run1 ...

----------------------------------------------------------------------

"""

from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# SBI imports
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference

# Optional plotting with corner
try:
    import corner
    _HAS_CORNER = True
except Exception:
    _HAS_CORNER = False


# ----------------------------
# Defaults (paper / notebook parameterization)
# ----------------------------
DEFAULT_PRIOR_LOW = [0.0, 0.0, 25.0, -3.0, 30.0, 29.0, 24.0, 25.0, 1.0]
DEFAULT_PRIOR_HIGH = [8.0, 0.3, 70.0, 2.0, 36.0, 38.0, 36.0, 36.0, 90.0]
DEFAULT_PARAM_LABELS = ['vA', 'cs', 'theta', 'rin', 'Stell', 'FUV', 'EUV', 'Xray', 'inclination']



# ----------------------------
# Configuration container
# ----------------------------

@dataclass
class Config:
    # Data paths
    sim_h5: str
    obs_txt: Optional[str]
    outdir: str

    # Device and reproducibility
    device: str
    seed: int

    # Observation cleaning window
    wave_low: float
    wave_high: float

    # Preprocess
    normalize_sims: bool
    obs_subtract_continuum: bool
    obs_normalize_by_max: bool
    drop_nan_rows: bool
    log10_params: List[int]

    # Training regressor

    # Parameter labels (for plots)
    labels: List[str]

    # Training regressor
    regressor: str  # "cnn" or "spender"
    test_frac: float
    batch_size: int
    lr: float
    epochs: int

    # SNR schedule for mock spectra generation
    snr0: float
    snr_tau: float
    snr_offset: float

    # CNN architecture
    cnn_channels: List[int]
    kernel_size: int
    latent_dim: int
    mlp_hidden: List[int]
    dropout: float

    # SBI setup
    prior_low: List[float]
    prior_high: List[float]
    maf_hidden_features: int
    maf_num_transforms: int
    n_posterior_samples: int

    # Diagnostics
    n_rank_samples: int  # posterior draws per test example for rank statistic


# ----------------------------
# I/O utilities
# ----------------------------

def load_sim_h5(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load simulation data from an HDF5 file that contains:
      - spec   (N, L)
      - wave   (L,)
      - params (N, P)
    """
    with h5py.File(path, "r") as f:
        spec = np.array(f["spec"])
        wave = np.array(f["wave"])
        params = np.array(f["params"])
    return wave, spec, params


def load_obs_txt(path: str) -> np.ndarray:
    """
    Load observation from a text file with columns:
      wave, flux, err
    """
    obs = np.genfromtxt(path)
    if obs.ndim != 2 or obs.shape[1] < 3:
        raise ValueError("Observation text file must be Nx3 (wave, flux, err).")
    return obs[:, :3]


# ----------------------------
# Preprocessing utilities
# ----------------------------

def drop_nan_rows(spec: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove rows where spectra or params contain NaN/Inf.
    """
    bad = np.any(~np.isfinite(spec), axis=1) | np.any(~np.isfinite(params), axis=1)
    return spec[~bad], params[~bad]


def minmax_per_row(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Min-max normalize each spectrum independently:
      x' = (x - min(x)) / (max(x) - min(x) + eps)
    """
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / (x_max - x_min + eps)


def apply_param_transforms(params: np.ndarray, log10_indices: List[int]) -> np.ndarray:
    """
    Apply parameter transforms (currently log10 on selected indices).
    This mirrors your notebook pattern, e.g. params[:,3] = log10(params[:,3]).
    """
    out = params.copy()
    for idx in log10_indices:
        out[:, idx] = np.log10(out[:, idx])
    return out


def clean_obs(
    obs: np.ndarray,
    wave_low: float,
    wave_high: float,
    subtract_continuum: bool = True,
    normalize_by_max: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean an observed spectrum:
    - cut to [wave_low, wave_high]
    - drop NaNs/Infs
    - subtract continuum (flux -= 1) if requested
    - normalize by max(|flux|) if requested
    Returns wave, flux, err arrays.
    """
    wave, flux, err = obs[:, 0], obs[:, 1], obs[:, 2]

    mask = (wave > wave_low) & (wave < wave_high)
    mask &= np.isfinite(flux) & np.isfinite(err)

    wave = wave[mask]
    flux = flux[mask]
    err = err[mask]

    if subtract_continuum:
        flux = flux - 1.0

    if normalize_by_max:
        denom = np.max(np.abs(flux))
        denom = denom if denom > 0 else 1.0
        flux = flux / denom
        err = err / denom

    return wave, flux, err


# ----------------------------
# Rebinning: simple trapezoid rebin
# ----------------------------

def centers2edges(x: np.ndarray) -> np.ndarray:
    """
    Convert bin centers -> bin edges assuming nearly uniform spacing.
    """
    x = np.asarray(x)
    dx = np.diff(x)
    edges = np.empty(len(x) + 1, dtype=np.float64)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def trapz_rebin(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    """
    Trapezoidal rebin of (x,y) onto new bin centers xnew.

    This is a practical, reasonably accurate rebinning method and is consistent
    in spirit with what you were doing in the notebooks (integrate then normalize).

    IMPORTANT: This assumes xnew range is within x range.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xnew = np.asarray(xnew, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Define edges for xnew bins.
    edges = centers2edges(xnew)

    # Safety check: x must cover edges range.
    if edges[0] < x[0] or edges[-1] > x[-1]:
        raise ValueError("xnew bin edges exceed the range of x. Provide compatible grids.")

    # For each xnew bin [edges[i], edges[i+1]], integrate y(x) dx and divide by bin width.
    out = np.zeros(len(xnew), dtype=np.float64)

    # We'll compute y(x) piecewise linear, integrate over overlaps.
    # This implementation is intentionally straightforward & readable.

    # Precompute segment slopes for piecewise linear interpolation.
    dx = np.diff(x)
    dy = np.diff(y)
    slopes = dy / dx

    # For each new bin:
    for i in range(len(xnew)):
        a, b = edges[i], edges[i + 1]

        # Find indices of x segments overlapping [a,b]
        # left segment index jl such that x[jl] <= a < x[jl+1]
        jl = np.searchsorted(x, a) - 1
        jr = np.searchsorted(x, b) - 1

        # Clamp indices to valid range
        jl = max(jl, 0)
        jr = min(jr, len(x) - 2)

        area = 0.0

        # Integrate over all overlapping segments
        for j in range(jl, jr + 1):
            seg_left = max(a, x[j])
            seg_right = min(b, x[j + 1])
            if seg_right <= seg_left:
                continue

            # y at seg_left via linear interpolation in segment j
            y_left = y[j] + slopes[j] * (seg_left - x[j])
            y_right = y[j] + slopes[j] * (seg_right - x[j])

            # trapezoid area on this sub-interval
            area += 0.5 * (y_left + y_right) * (seg_right - seg_left)

        out[i] = area / (b - a)

    return out


# ----------------------------
# Mock spectra generation + SNR schedule
# ----------------------------

def snr_schedule_exp_decay(epoch: int, snr0: float, tau: float, offset: float) -> float:
    """
    Exponential decay SNR schedule:
      snr(epoch) = snr0 * exp(-epoch/tau) + offset
    Mirrors your notebook idea of gradually changing noise regime.
    """
    return float(snr0 * math.exp(-epoch / tau) + offset)


def gen_mock_spectra(
    wave_sim: np.ndarray,
    specs_sim: np.ndarray,
    wave_obs: np.ndarray,
    err_obs: np.ndarray,
    snr: float,
    seed: int
) -> np.ndarray:
    """
    Rebin simulated spectra from wave_sim to wave_obs and add Gaussian noise.

    Noise model:
      sigma_i = err_obs_i / snr
    This matches the spirit of your notebook: noise tied to observation errors and SNR.

    Returns mock spectra array with shape (N, len(wave_obs)).
    """
    rng = np.random.default_rng(seed)

    Lobs = len(wave_obs)
    N = specs_sim.shape[0]
    out = np.zeros((N, Lobs), dtype=np.float64)

    # Avoid zero/negative snr
    snr = max(float(snr), 1e-6)
    sigma = err_obs / snr

    for i in range(N):
        reb = trapz_rebin(wave_sim, specs_sim[i], wave_obs)
        noise = rng.normal(0.0, sigma, size=reb.shape)
        out[i] = reb + noise

    return out


# ----------------------------
# Regressor models
# ----------------------------

class CNNRegressor1D(nn.Module):
    """
    A simple 1D CNN regressor:
      spectrum (B,L) -> CNN -> global avg pool -> MLP -> (B,P)

    This is the default "no external dependency" option.
    """
    def __init__(
        self,
        L: int,
        P: int,
        cnn_channels: List[int],
        kernel_size: int,
        latent_dim: int,
        mlp_hidden: List[int],
        dropout: float
    ):
        super().__init__()

        k = int(kernel_size)
        pad = k // 2

        layers = []
        c_in = 1
        for c_out in cnn_channels:
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            c_in = c_out
        self.cnn = nn.Sequential(*layers)

        # Infer dimension after CNN by running a dummy tensor through.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, L)
            h = self.cnn(dummy)
            pooled = h.mean(dim=-1)  # global average pool
            d = pooled.shape[1]

        self.to_latent = nn.Sequential(
            nn.Linear(d, latent_dim),
            nn.ReLU()
        )

        head = []
        in_dim = latent_dim
        for hdim in mlp_hidden:
            head += [nn.Linear(in_dim, hdim), nn.ReLU()]
            if dropout > 0:
                head += [nn.Dropout(dropout)]
            in_dim = hdim
        head += [nn.Linear(in_dim, P)]
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L)
        x = x.unsqueeze(1)          # (B,1,L)
        h = self.cnn(x)             # (B,C,L')
        pooled = h.mean(dim=-1)     # (B,C)
        z = self.to_latent(pooled)  # (B,latent)
        y = self.head(z)            # (B,P)
        return y


class SpenderRegressor(nn.Module):
    """
    A wrapper regressor that uses spender.SpectrumEncoder if available.

    spender typically expects input shaped like (B, L) or (B, 1, L) depending
    on version; we handle a common case here.

    If spender is not installed, constructing this will fail.
    """
    def __init__(self, L: int, P: int, latent_dim: int, mlp_hidden: List[int], dropout: float):
        super().__init__()

        # Import spender lazily so script can still run without spender installed.
        try:
            from spender import SpectrumEncoder
        except Exception as e:
            raise ImportError(
                "spender is not installed or cannot be imported. "
                "Install spender or use --regressor cnn."
            ) from e

        # Create an encoder. Exact signature may vary across spender versions.
        # We keep it flexible with basic arguments.
        self.encoder = SpectrumEncoder(n_latent=latent_dim)

        # MLP head maps latent -> params
        head = []
        in_dim = latent_dim
        for hdim in mlp_hidden:
            head += [nn.Linear(in_dim, hdim), nn.ReLU()]
            if dropout > 0:
                head += [nn.Dropout(dropout)]
            in_dim = hdim
        head += [nn.Linear(in_dim, P)]
        self.head = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected (B,L)
        # spender encoder usually wants (B,L); if it wants (B,1,L) you can adjust here.
        z = self.encoder(x)
        y = self.head(z)
        return y


# ----------------------------
# Training utilities (regressor)
# ----------------------------

def train_regressor(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int
) -> Tuple[nn.Module, dict]:
    """
    Standard regression training loop with MSE loss.

    We keep it simple and robust:
    - SGD + momentum (like your notebook flavor)
    - Save the best validation checkpoint in memory
    """
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        # ----- Train -----
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # ----- Validate -----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Keep best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[Regressor] epoch {epoch+1:03d}/{epochs}  train={train_loss:.6g}  val={val_loss:.6g}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def plot_losses(history: dict, outpath: str):
    """
    Save regressor training curves.
    """
    plt.figure()
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# SBI utilities
# ----------------------------

def build_prior(prior_low: List[float], prior_high: List[float], device: str):
    """
    Construct a BoxUniform prior used by SBI SNPE.
    """
    low = torch.tensor(prior_low, dtype=torch.float32, device=device)
    high = torch.tensor(prior_high, dtype=torch.float32, device=device)
    return sbi_utils.BoxUniform(low=low, high=high, device=device)


def train_snpe(
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    prior,
    device: str,
    maf_hidden_features: int,
    maf_num_transforms: int
):
    """
    Train SNPE with a MAF density estimator to learn p(theta | x).

    Here, x is NOT the raw spectrum; x is the regressor output \hat{theta}.
    So we learn p(theta_true | theta_hat).
    """
    density_estimator = sbi_utils.posterior_nn(
        "maf",
        hidden_features=int(maf_hidden_features),
        num_transforms=int(maf_num_transforms)
    )

    snpe = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator, device=device)
    snpe.append_simulations(theta_train, x_train)
    estimator = snpe.train()
    posterior = snpe.build_posterior(estimator)
    return posterior, estimator


@torch.no_grad()
def compute_rank_stats(
    posterior,
    theta_test: torch.Tensor,
    x_test: torch.Tensor,
    n_post: int
) -> np.ndarray:
    """
    Compute SBC-style rank statistics:
      For each test sample i and parameter j:
        rank_ij = #{ posterior_samples_ij < theta_true_ij }

    If the posterior is calibrated, the rank histogram should be ~uniform.

    Returns:
      ranks: (Ntest, P) integer array in [0, n_post]
    """
    theta_np = theta_test.detach().cpu().numpy()
    ranks = []

    for i in tqdm(range(theta_np.shape[0]), desc="Rank calibration"):
        # Sample posterior conditioned on x_test[i]
        samp = posterior.sample((n_post,), x=x_test[i], show_progress_bars=False).cpu().numpy()
        # Count how many samples are less than the truth for each parameter
        r = np.sum(samp < theta_np[i][None, :], axis=0)
        ranks.append(r)

    return np.asarray(ranks, dtype=np.int64)


def plot_rank_hist(ranks: np.ndarray, outpath: str, n_post: int, labels: Optional[List[str]] = None):
    """
    Plot rank histograms (one per parameter). Uniform ~ good calibration.
    """
    N, P = ranks.shape
    if labels is None:
        labels = [f"theta_{i}" for i in range(P)]

    ncols = 3
    nrows = int(np.ceil(P / ncols))
    plt.figure(figsize=(4 * ncols, 3 * nrows))

    for j in range(P):
        ax = plt.subplot(nrows, ncols, j + 1)
        ax.hist(ranks[:, j] / float(n_post), range=(0, 1), density=True, histtype="step", linewidth=2)
        ax.plot([0, 1], [1, 1], "k--", linewidth=1)
        ax.set_title(labels[j])
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_corner(samples: np.ndarray, outpath: str, labels: Optional[List[str]] = None, truths: Optional[np.ndarray] = None):
    """
    Save a corner plot of posterior samples (requires corner).
    """
    if not _HAS_CORNER:
        print("[WARN] corner is not installed; skipping corner plot.")
        return

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        bins=25,
        smooth=1,
        fill_contours=True,
        show_titles=True,
        quantiles=(0.16, 0.5, 0.84),
        plot_datapoints=False,
        hist_kwargs={"density": True},
    )
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ----------------------------
# Main pipeline
# ----------------------------

def main(cfg: Config):
    # ----------------------------
    # Setup
    # ----------------------------
    os.makedirs(cfg.outdir, exist_ok=True)

    # Reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device selection (fallback to CPU if CUDA unavailable)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {cfg.outdir}")

    # ----------------------------
    # Load simulations
    # ----------------------------
    wave_sim, specs_sim, params = load_sim_h5(cfg.sim_h5)

    # Basic shape checks
    if specs_sim.ndim != 2:
        raise ValueError("spec dataset must be 2D: (N, L).")
    if params.ndim != 2:
        raise ValueError("params dataset must be 2D: (N, P).")

    N, L = specs_sim.shape
    P = params.shape[1]
    print(f"[INFO] Loaded simulations: N={N}, L={L}, P={P}")

    # ----------------------------
    # Preprocess simulations
    # ----------------------------
    if cfg.drop_nan_rows:
        specs_sim, params = drop_nan_rows(specs_sim, params)
        print(f"[INFO] After dropping NaN rows: N={specs_sim.shape[0]}")

    if cfg.normalize_sims:
        specs_sim = minmax_per_row(specs_sim)

    if len(cfg.log10_params) > 0:
        params = apply_param_transforms(params, cfg.log10_params)

    # ----------------------------
    # Load and clean observation (optional)
    # ----------------------------
    # If no observation is provided, we will:
    # - use sim wave grid as the "obs grid"
    # - define a constant error array
    # - use a held-out simulation spectrum as a stand-in "observation"
    if cfg.obs_txt is not None:
        obs = load_obs_txt(cfg.obs_txt)
        wave_obs, flux_obs, err_obs = clean_obs(
            obs,
            wave_low=cfg.wave_low,
            wave_high=cfg.wave_high,
            subtract_continuum=cfg.obs_subtract_continuum,
            normalize_by_max=cfg.obs_normalize_by_max
        )
        print(f"[INFO] Loaded observation: Nobs={len(wave_obs)}")
    else:
        wave_obs = wave_sim.copy()
        err_obs = np.ones_like(wave_obs) * 0.01
        flux_obs = None
        print("[INFO] No observation provided; will use a held-out simulation spectrum as a proxy.")

    # ----------------------------
    # Train/test split on simulations
    # ----------------------------
    N = specs_sim.shape[0]
    idx = np.random.permutation(N)
    n_test = int(cfg.test_frac * N)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    specs_train = specs_sim[train_idx]
    params_train = params[train_idx]
    specs_test = specs_sim[test_idx]
    params_test = params[test_idx]

    print(f"[INFO] Train size: {len(train_idx)}, Test size: {len(test_idx)}")

    # ----------------------------
    # Generate mock noisy spectra on obs grid
    # ----------------------------
    # We need an observation grid + error model to generate realistic mocks.
    # If you provided a real obs: use its wave grid and error vector.
    # Otherwise: use sim wave grid and constant errors.

    # For validation, we keep a fixed mock set with SNR=1 to stabilize comparison.
    val_snr = 1.0
    x_val_np = gen_mock_spectra(
        wave_sim, specs_test, wave_obs, err_obs,
        snr=val_snr, seed=cfg.seed + 999
    )

    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(params_test, dtype=torch.float32)

    # ----------------------------
    # Build regressor model
    # ----------------------------
    if cfg.regressor.lower() == "cnn":
        reg = CNNRegressor1D(
            L=len(wave_obs),
            P=P,
            cnn_channels=cfg.cnn_channels,
            kernel_size=cfg.kernel_size,
            latent_dim=cfg.latent_dim,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout
        )
    elif cfg.regressor.lower() == "spender":
        reg = SpenderRegressor(
            L=len(wave_obs),
            P=P,
            latent_dim=cfg.latent_dim,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout
        )
    else:
        raise ValueError("--regressor must be 'cnn' or 'spender'")

    # ----------------------------
    # Train regressor with an SNR schedule
    # ----------------------------
    # In your notebook family B, you were effectively training on mock/noisy spectra
    # tied to the observation noise model. Here we regenerate training mocks per epoch
    # with a decaying SNR schedule.

    # We'll do this by:
    # - generating a fresh mock dataset each epoch
    # - training for ONE epoch on that dataset (warm-starting weights)
    # This mimics "noise annealing" behavior without complex code.
    reg = reg.to(device)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    # Initialize optimizer once so learning continues across epochs
    opt = torch.optim.SGD(reg.parameters(), lr=cfg.lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    # Fixed validation loader
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    for epoch in range(cfg.epochs):
        # Compute SNR for this epoch
        snr = snr_schedule_exp_decay(epoch, cfg.snr0, cfg.snr_tau, cfg.snr_offset)

        # Generate fresh training mocks at this SNR
        x_train_np = gen_mock_spectra(
            wave_sim, specs_train, wave_obs, err_obs,
            snr=snr, seed=cfg.seed + epoch
        )
        x_train = torch.tensor(x_train_np, dtype=torch.float32)
        y_train = torch.tensor(params_train, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True
        )

        # ----- One epoch of training -----
        reg.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = reg(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # ----- Validation -----
        reg.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = reg(xb)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in reg.state_dict().items()}

        print(f"[Regressor] epoch {epoch+1:03d}/{cfg.epochs}  snr={snr:.3g}  train={train_loss:.6g}  val={val_loss:.6g}")

    if best_state is not None:
        reg.load_state_dict(best_state)

    # Save regressor weights and loss plot
    torch.save(reg.state_dict(), os.path.join(cfg.outdir, "regressor.pt"))
    plot_losses(history, os.path.join(cfg.outdir, "regressor_losses.png"))

    # ----------------------------
    # Build SBI dataset: (theta_true, theta_hat)
    # ----------------------------
    # We create a single large mock set at a fixed SNR (=1) and compute theta_hat = reg(mock_spectrum).
    # Then SNPE learns p(theta_true | theta_hat).
    reg.eval()
    with torch.no_grad():
        specs_mock = gen_mock_spectra(
            wave_sim, specs_sim, wave_obs, err_obs,
            snr=1.0, seed=cfg.seed + 202
        )
        X_hat = reg(torch.tensor(specs_mock, dtype=torch.float32, device=device)).detach().cpu()

    theta_true = torch.tensor(params, dtype=torch.float32)  # on CPU for now

    # Split for SNPE training/testing (for rank diagnostic)
    N2 = X_hat.shape[0]
    perm2 = torch.randperm(N2)
    cut2 = int(0.9 * N2)
    tr2 = perm2[:cut2]
    te2 = perm2[cut2:]

    theta_train = theta_true[tr2].to(device)
    x_train = X_hat[tr2].to(device)

    theta_test = theta_true[te2].to(device)
    x_test = X_hat[te2].to(device)

    # ----------------------------
    # Train SNPE (MAF) posterior
    # ----------------------------
    if len(cfg.prior_low) != P or len(cfg.prior_high) != P:
        raise ValueError(
            f"Prior bounds dimension mismatch: got low/high of length "
            f"{len(cfg.prior_low)}/{len(cfg.prior_high)} but params have P={P}."
        )

    prior = build_prior(cfg.prior_low, cfg.prior_high, device=device)
    posterior, estimator = train_snpe(
        theta_train=theta_train,
        x_train=x_train,
        prior=prior,
        device=device,
        maf_hidden_features=cfg.maf_hidden_features,
        maf_num_transforms=cfg.maf_num_transforms
    )

    # Save SNPE estimator weights
    torch.save(estimator.state_dict(), os.path.join(cfg.outdir, "snpe_estimator.pt"))

    # ----------------------------
    # Rank diagnostic (SBC-style)
    # ----------------------------
    ranks = compute_rank_stats(posterior, theta_test, x_test, n_post=cfg.n_rank_samples)
    np.save(os.path.join(cfg.outdir, "ranks.npy"), ranks)
    plot_rank_hist(ranks, os.path.join(cfg.outdir, "rank_hist.png"), n_post=cfg.n_rank_samples, labels=list(cfg.labels) if hasattr(cfg,"labels") and len(cfg.labels)==P else None)

    # ----------------------------
    # Inference on observation (or proxy)
    # ----------------------------
    # If no obs was provided, choose a test spectrum as proxy "observation".
    if flux_obs is None:
        flux_obs = specs_test[0].copy()
        # If we are in "no obs" mode, flux_obs is already on wave_obs (= wave_sim)
        # and has already been minmax normalized if cfg.normalize_sims=True.
        # Define err_obs (already constant) and proceed.

    # Run regressor to get theta_hat_obs
    reg.eval()
    with torch.no_grad():
        x_obs_t = torch.tensor(flux_obs[None, :], dtype=torch.float32, device=device)
        theta_hat_obs = reg(x_obs_t).detach()

    # Sample posterior p(theta | theta_hat_obs)
    with torch.no_grad():
        samples = posterior.sample(
            (cfg.n_posterior_samples,),
            x=theta_hat_obs,
            show_progress_bars=False
        ).cpu().numpy()

    np.save(os.path.join(cfg.outdir, "theta_hat_obs.npy"), theta_hat_obs.cpu().numpy())
    np.save(os.path.join(cfg.outdir, "posterior_samples_obs.npy"), samples)

    # Corner plot (optional)
    labels = list(cfg.labels) if hasattr(cfg,"labels") and len(cfg.labels)==P else [f"theta_{i}" for i in range(P)]
    plot_corner(samples, os.path.join(cfg.outdir, "posterior_corner.png"), labels=labels, truths=None)

        # ----------------------------
    # Convenience: copy key plots into ./figures for README embedding
    # ----------------------------
    fig_dir = os.path.join(cfg.outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    try:
        import shutil
        shutil.copyfile(os.path.join(cfg.outdir, "regressor_losses.png"),
                        os.path.join(fig_dir, "fig0_regressor_losses.png"))
        shutil.copyfile(os.path.join(cfg.outdir, "posterior_corner.png"),
                        os.path.join(fig_dir, "fig2_posterior_corner.png"))
        shutil.copyfile(os.path.join(cfg.outdir, "rank_hist.png"),
                        os.path.join(fig_dir, "fig4_sbc_rank_hist.png"))
    except Exception:
        pass

    print("[DONE] Outputs saved to:", cfg.outdir)


# ----------------------------
# Argument parsing
# ----------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Single-file Family-B pipeline: regressor summaries + SNPE calibrated posterior."
    )

    # Required
    p.add_argument("--sim_h5", type=str, required=True, help="Path to simulation HDF5 file.")
    p.add_argument("--outdir", type=str, required=True, help="Output directory.")

    # Optional observation
    p.add_argument("--obs_txt", type=str, default=None, help="Optional observation text file (wave,flux,err).")
    p.add_argument("--wave_low", type=float, default=-400.0, help="Lower wave/vel cut for observation.")
    p.add_argument("--wave_high", type=float, default=300.0, help="Upper wave/vel cut for observation.")

    # Repro/device
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Preprocess
    p.add_argument("--normalize_sims", action="store_true", help="Min-max normalize simulated spectra per spectrum.")
    p.add_argument("--no_normalize_sims", action="store_true", help="Disable simulation normalization (overrides).")
    p.add_argument("--drop_nan_rows", action="store_true", help="Drop NaN/Inf rows.")
    p.add_argument("--no_drop_nan_rows", action="store_true", help="Disable NaN dropping (overrides).")
    p.add_argument("--log10_params", type=int, nargs="*", default=[],
                   help="Parameter indices to log10-transform (e.g. --log10_params 3).")

    # Observation cleaning flags
    p.add_argument("--obs_subtract_continuum", action="store_true", help="Subtract 1 from observation flux.")
    p.add_argument("--no_obs_subtract_continuum", action="store_true", help="Disable continuum subtraction.")
    p.add_argument("--obs_normalize_by_max", action="store_true", help="Normalize observation by max(|flux|).")
    p.add_argument("--no_obs_normalize_by_max", action="store_true", help="Disable obs normalization.")

    # Regressor choice
    p.add_argument("--regressor", type=str, default="cnn", choices=["cnn", "spender"],
                   help="Which regressor to use: cnn or spender.")

    p.add_argument("--labels", type=str, nargs="*", default=DEFAULT_PARAM_LABELS,
                   help="Optional parameter labels (P strings). Used in plots.")

    # Training hyperparams
    p.add_argument("--test_frac", type=float, default=0.15, help="Fraction for test split.")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for regressor training.")
    p.add_argument("--lr", type=float, default=5e-3, help="Learning rate.")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs (for regressor).")

    # SNR schedule
    p.add_argument("--snr0", type=float, default=60.0, help="Initial SNR for mock training data.")
    p.add_argument("--snr_tau", type=float, default=10.0, help="Decay timescale for SNR schedule.")
    p.add_argument("--snr_offset", type=float, default=-1.8, help="Offset for SNR schedule.")

    # CNN architecture
    p.add_argument("--cnn_channels", type=int, nargs="*", default=[16, 32, 64], help="CNN channel sizes.")
    p.add_argument("--kernel_size", type=int, default=7, help="CNN kernel size.")
    p.add_argument("--latent_dim", type=int, default=128, help="Latent size before MLP head.")
    p.add_argument("--mlp_hidden", type=int, nargs="*", default=[128, 128], help="MLP hidden sizes.")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout in the MLP head.")

    # SBI params
    p.add_argument("--prior_low", type=float, nargs="+", default=DEFAULT_PRIOR_LOW,
                   help="Prior lower bounds (length P). Defaults match paper/notebooks.")
    p.add_argument("--prior_high", type=float, nargs="+", default=DEFAULT_PRIOR_HIGH,
                   help="Prior upper bounds (length P). Defaults match paper/notebooks.")
    p.add_argument("--maf_hidden_features", type=int, default=50, help="MAF hidden features.")
    p.add_argument("--maf_num_transforms", type=int, default=10, help="MAF number of transforms.")
    p.add_argument("--n_posterior_samples", type=int, default=10000, help="Posterior samples for final inference.")

    # Diagnostics
    p.add_argument("--n_rank_samples", type=int, default=5000, help="Posterior draws per test for rank statistic.")

    args = p.parse_args()

    # Handle overriding flags cleanly
    normalize_sims = True
    if args.no_normalize_sims:
        normalize_sims = False
    elif args.normalize_sims:
        normalize_sims = True

    drop_nan = True
    if args.no_drop_nan_rows:
        drop_nan = False
    elif args.drop_nan_rows:
        drop_nan = True

    obs_sub_cont = True
    if args.no_obs_subtract_continuum:
        obs_sub_cont = False
    elif args.obs_subtract_continuum:
        obs_sub_cont = True

    obs_norm_max = True
    if args.no_obs_normalize_by_max:
        obs_norm_max = False
    elif args.obs_normalize_by_max:
        obs_norm_max = True

    cfg = Config(
        sim_h5=args.sim_h5,
        obs_txt=args.obs_txt,
        outdir=args.outdir,
        device=args.device,
        seed=args.seed,

        wave_low=args.wave_low,
        wave_high=args.wave_high,

        normalize_sims=normalize_sims,
        obs_subtract_continuum=obs_sub_cont,
        obs_normalize_by_max=obs_norm_max,
        drop_nan_rows=drop_nan,
        log10_params=list(args.log10_params),

        regressor=args.regressor,

        labels=list(args.labels),

        regressor=args.regressor,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,

        snr0=args.snr0,
        snr_tau=args.snr_tau,
        snr_offset=args.snr_offset,

        cnn_channels=list(args.cnn_channels),
        kernel_size=args.kernel_size,
        latent_dim=args.latent_dim,
        mlp_hidden=list(args.mlp_hidden),
        dropout=args.dropout,

        prior_low=list(args.prior_low),
        prior_high=list(args.prior_high),
        maf_hidden_features=args.maf_hidden_features,
        maf_num_transforms=args.maf_num_transforms,
        n_posterior_samples=args.n_posterior_samples,

        n_rank_samples=args.n_rank_samples
    )

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
