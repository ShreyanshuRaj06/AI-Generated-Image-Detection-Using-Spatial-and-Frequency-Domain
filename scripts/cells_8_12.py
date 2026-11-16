# [MARKDOWN]
## Cell 8 — Dataset & DataLoaders

# ===CELL_BREAK===

# [CODE]
"""
AIImageDataset — Spatial/Frequency Domain AI-Image Detection
=============================================================

Sampling Protocols
------------------
Protocol N  (native):   Each dataset is used at its own native resolution.
                        No spatial resizing is ever applied.  The model receives
                        crops of whatever size was stored in the manifest.

Protocol C  (common):   All images are additionally center-cropped to a fixed
                        ``common_crop`` side length (default 32 px) BEFORE
                        being passed to the model.  This equalises resolution
                        across heterogeneous datasets so that cross-dataset
                        comparisons are resolution-fair.

Why NO resize?
--------------
Resizing (bilinear / bicubic / lanczos) is a **low-pass filter** in the
spatial domain.  In the frequency domain it:
  * Attenuates or aliases high-frequency components that carry detector-relevant
    artefacts (GAN grid patterns, diffusion-model ringing, JPEG recompression
    peaks).
  * Introduces its own spectral signature that may either leak into or mask the
    forensic signal.
Cropping is a *lossless* window operation: the spectrum of the crop is the
windowed DTFT of the original — high-frequency content is preserved.

Why cropping still does NOT remove the resolution shortcut?
-----------------------------------------------------------
If a model is trained on real images at a systematically different resolution
than AI images (a common dataset construction artefact), even a fixed-size
center-crop retains the same spectral statistics, because the underlying pixel
density and noise floor of the original capture / generation pipeline are
unchanged.  Protocol C *reduces* but does not *eliminate* the shortcut; the
shortcut is fully removed only by training on resolution-matched pairs or by
explicit resolution-stratified evaluation.
"""

import os
import csv
import math
import time
import json
import random
import logging
import warnings
import itertools
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AIImageDataset(Dataset):
    """
    Stream-from-disk dataset for AI vs. real image detection.

    Parameters
    ----------
    manifest_path : str
        Path to a CSV file with at least columns: ``path``, ``label``, ``split``.
        ``path`` may be absolute or relative to *subset_root*.
        ``label``: 0 = real, 1 = AI-generated.
    subset_root : str
        Absolute base directory prepended to relative paths in the manifest.
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    max_samples_per_class : Optional[int]
        Class-balanced cap per class. Subsampled with a fixed seed.
    common_crop : Optional[int]
        Protocol C: PIL center-crop every image to this side length after
        loading (NO interpolation — pure pixel window).  None → Protocol N.
    augment : bool
        If True (and split == 'train'), apply random horizontal flip and
        light colour jitter.  Augmentation is intentionally conservative to
        avoid introducing spectral artefacts the frequency branch would detect.
    cfg : Optional[Any]
        Global config object used for the random seed.
    """

    _JITTER_BRIGHTNESS = 0.1
    _JITTER_CONTRAST   = 0.1
    _JITTER_SATURATION = 0.1
    _JITTER_HUE        = 0.02

    def __init__(
        self,
        manifest_path: str,
        subset_root: str,
        split: str,
        max_samples_per_class: Optional[int] = 20_000,
        common_crop: Optional[int] = None,
        augment: bool = False,
        cfg: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.subset_root = Path(subset_root)
        self.split       = split
        self.common_crop = common_crop
        self.augment     = augment and (split == "train")
        self._seed       = getattr(cfg, "seeds", [42])[0] if cfg is not None else 42

        # ---- load manifest ------------------------------------------------
        df = pd.read_csv(manifest_path)
        required_cols = {"path", "label", "split"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")

        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}"
            )

        df["label"] = df["label"].astype(int)

        # ---- class balancing ----------------------------------------------
        if max_samples_per_class is not None:
            rng = random.Random(self._seed)
            balanced_parts: List[pd.DataFrame] = []
            for lbl, grp in df.groupby("label"):
                if len(grp) > max_samples_per_class:
                    idx = rng.sample(range(len(grp)), max_samples_per_class)
                    grp = grp.iloc[idx].reset_index(drop=True)
                balanced_parts.append(grp)
            df = pd.concat(balanced_parts, ignore_index=True)

        if split == "train":
            df = df.sample(frac=1, random_state=self._seed).reset_index(drop=True)

        self._samples: List[Tuple[Path, int]] = [
            (self._resolve(row["path"]), int(row["label"]))
            for _, row in df.iterrows()
        ]

        # ---- build transforms ---------------------------------------------
        aug_transforms: List[Any] = []
        if self.augment:
            aug_transforms += [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=self._JITTER_BRIGHTNESS,
                    contrast=self._JITTER_CONTRAST,
                    saturation=self._JITTER_SATURATION,
                    hue=self._JITTER_HUE,
                ),
            ]
        # ToTensor: PIL [0,255] uint8 → float32 [0,1]
        # NOTE: NO ImageNet normalisation here; normalisation lives INSIDE SpatialBranch.
        self._transform = T.Compose(aug_transforms + [T.ToTensor()])

        logger.info(
            "AIImageDataset | split=%s | n=%d | augment=%s | protocol=%s",
            split,
            len(self._samples),
            self.augment,
            f"C(crop={common_crop})" if common_crop else "N",
        )

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else self.subset_root / p

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self._samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            logger.warning("Corrupted (UnidentifiedImageError): %s — using black fallback", img_path)
            img   = self._black_fallback()
            label = -1  # sentinel; callers filter label == -1
        except Exception as exc:
            logger.warning("Failed to load %s: %s — using black fallback", img_path, exc)
            img   = self._black_fallback()
            label = -1

        # Protocol C: center-crop BEFORE ToTensor (pure pixel window, no interp)
        if self.common_crop is not None:
            w, h = img.size
            if w >= self.common_crop and h >= self.common_crop:
                img = TF.center_crop(img, self.common_crop)
            else:
                # Pad with reflection then crop if image is smaller than target
                pad_h = max(0, self.common_crop - h)
                pad_w = max(0, self.common_crop - w)
                img = TF.pad(
                    img,
                    [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2],
                    padding_mode="reflect",
                )
                img = TF.center_crop(img, self.common_crop)

        tensor = self._transform(img)  # float32 [0,1] in [C,H,W]
        return tensor, label

    @staticmethod
    def _black_fallback(size: int = 32) -> Image.Image:
        return Image.new("RGB", (size, size), color=0)


# ---------------------------------------------------------------------------
# DataLoaders factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    manifest_path: str,
    subset_root: str,
    cfg: Any,
    protocol: str = "N",
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a manifest CSV.

    Parameters
    ----------
    manifest_path : str
        Path to the manifest CSV (columns: path, label, split).
    subset_root : str
        Root directory for relative image paths.
    cfg : CFG
        Global config (batch_size, num_workers, pin_memory,
        persistent_workers, common_crop, seeds).
    protocol : str
        ``'N'`` → native resolution (no extra crop).
        ``'C'`` → center-crop to ``cfg.common_crop`` px.
    augment_train : bool
        Whether to apply augmentation on the training split.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    if protocol not in ("N", "C"):
        raise ValueError(f"Unknown protocol '{protocol}'. Choose 'N' or 'C'.")

    common_crop = getattr(cfg, "common_crop", 32) if protocol == "C" else None
    n_workers   = getattr(cfg, "num_workers", 2)
    is_quick    = getattr(cfg, "QUICK_TEST", False)
    max_per_cls = (
        getattr(cfg, "quick_test_samples", 500)
        if is_quick
        else getattr(cfg, "max_samples_per_class", 20_000)
    )

    _ds_kwargs = dict(
        manifest_path=manifest_path,
        subset_root=subset_root,
        common_crop=common_crop,
        cfg=cfg,
        max_samples_per_class=max_per_cls,
    )
    _loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=n_workers,
        pin_memory=getattr(cfg, "pin_memory", True),
        persistent_workers=getattr(cfg, "persistent_workers", True) and n_workers > 0,
    )

    train_ds = AIImageDataset(**_ds_kwargs, split="train", augment=augment_train)
    val_ds   = AIImageDataset(**_ds_kwargs, split="val",   augment=False)
    test_ds  = AIImageDataset(**_ds_kwargs, split="test",  augment=False)

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_ds,  shuffle=False, drop_last=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, drop_last=False, **_loader_kwargs)

    return train_loader, val_loader, test_loader


print("✅ Cell 8: AIImageDataset + get_dataloaders defined.")

# ===CELL_BREAK===

# [MARKDOWN]
## Cell 9 — Model Architecture

# ===CELL_BREAK===

# [CODE]
"""
Cell 9 — Model: SpatialBranch, FrequencyBranch, FusedDetector
"""

import torchvision.models as tvm

# ---------------------------------------------------------------------------
# Spatial Branch
# ---------------------------------------------------------------------------

class SpatialBranch(nn.Module):
    """
    Pretrained CNN backbone (ResNet-18 or MobileNetV2) with the classification
    head removed.  ImageNet normalisation is applied **inside** this module so
    the rest of the pipeline sees raw [0, 1] tensors.

    Output feature dimensions:
        ResNet-18   → 512
        MobileNetV2 → 1280

    Design note: Global average pooling makes the branch resolution-agnostic,
    so it handles both 224×224 and 32×32 inputs without resizing.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone

        weights_arg = "IMAGENET1K_V1" if pretrained else None

        if backbone == "resnet18":
            base = tvm.resnet18(weights=weights_arg)
            self.out_dim  = base.fc.in_features          # 512
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self.pool = nn.AdaptiveAvgPool2d(1)

        elif backbone == "mobilenet_v2":
            base = tvm.mobilenet_v2(weights=weights_arg)
            self.out_dim  = base.classifier[1].in_features   # 1280
            self.features = base.features
            self.pool     = nn.AdaptiveAvgPool2d(1)

        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                "Choose 'resnet18' or 'mobilenet_v2'."
            )

        # Registered as buffers so they move with .to(device)
        self.register_buffer(
            "_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std",  torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, 3, H, W] float32 in [0, 1] → [B, out_dim]"""
        x     = (x - self._mean) / self._std
        feats = self.features(x)
        feats = self.pool(feats)
        return feats.flatten(1)


# ---------------------------------------------------------------------------
# Frequency Branch
# ---------------------------------------------------------------------------

class FrequencyBranch(nn.Module):
    """
    Operates on raw [0, 1] grayscale images in the spectral domain.

    Modes
    -----
    'fft' : 2-D FFT (torch.fft.rfft2 + zero-pad to H×W) → log-magnitude.
    'dct' : 2-D DCT-II (orthonormal, separable) → log-magnitude.

    DCT Implementation
    ------------------
    Orthonormal DCT-II matrix D ∈ ℝ^{N×N}:
        D[k,n] = s_k · cos(π(2n+1)k / 2N)
        s_0 = √(1/N),  s_k = √(2/N) for k>0
    2-D DCT = D·x·Dᵀ  (separable; never builds N²×N² Kronecker product).
    D registered as a buffer; rebuilt on-demand per new N; cached in a dict.

    Hann window
    -----------
    Optional 2-D Hann window (outer product of two 1-D Hann windows) applied
    to the spatial signal BEFORE the transform to suppress spectral leakage
    caused by the implicit rectangular window of the crop boundary.

    Per-image normalisation
    -----------------------
    After log(1+|spectrum|), each image's spectrum is min-max normalised
    independently so that the small CNN input is always in [0, 1] regardless
    of image content or transform scale.
    """

    def __init__(
        self,
        freq_mode: str = "fft",
        use_hann:  bool = True,
        out_dim:   int  = 256,
    ) -> None:
        super().__init__()
        if freq_mode not in ("fft", "dct"):
            raise ValueError(f"freq_mode must be 'fft' or 'dct', got '{freq_mode}'.")
        self.freq_mode = freq_mode
        self.use_hann  = use_hann
        self.out_dim   = out_dim

        # Caches keyed by spatial size (avoid re-allocating on each forward pass)
        self._dct_cache:  Dict[int, torch.Tensor]        = {}
        self._hann_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # 4 conv blocks; AdaptiveAvgPool2d(1) gives resolution-agnostic output
        self.cnn = nn.Sequential(
            _conv_block(1, 32),
            nn.MaxPool2d(2, 2),
            _conv_block(32, 64),
            nn.MaxPool2d(2, 2),
            _conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            _conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, out_dim)

    # -----------------------------------------------------------------------
    # Buffer helpers
    # -----------------------------------------------------------------------

    def _build_dct_matrix(self, N: int, device: torch.device) -> torch.Tensor:
        """Orthonormal DCT-II matrix D of shape [N, N]. Cached per N."""
        if N in self._dct_cache:
            return self._dct_cache[N].to(device)
        k = torch.arange(N, dtype=torch.float64, device=device).unsqueeze(1)
        n = torch.arange(N, dtype=torch.float64, device=device).unsqueeze(0)
        D = torch.cos(math.pi * (2 * n + 1) * k / (2 * N))
        scale    = torch.full((N,), math.sqrt(2.0 / N), dtype=torch.float64, device=device)
        scale[0] = math.sqrt(1.0 / N)
        D = (D * scale.unsqueeze(1)).float()
        self._dct_cache[N] = D.cpu()
        return D

    def _build_hann_window(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """2-D Hann window [1,1,H,W] = outer(hann_H, hann_W). Cached per (H,W)."""
        key = (H, W)
        if key in self._hann_cache:
            return self._hann_cache[key].to(device)
        h1d   = torch.hann_window(H, periodic=False, device=device)
        w1d   = torch.hann_window(W, periodic=False, device=device)
        win2d = torch.outer(h1d, w1d).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        self._hann_cache[key] = win2d.cpu()
        return win2d

    # -----------------------------------------------------------------------
    # Spectrum computation
    # -----------------------------------------------------------------------

    def _compute_spectrum(self, gray: torch.Tensor) -> torch.Tensor:
        """
        gray : [B, 1, H, W] in [0,1]
        Returns [B, 1, H, W] log(1+|spectrum|), min-max normalised per image.
        """
        B, _, H, W = gray.shape
        device = gray.device

        if self.use_hann:
            win  = self._build_hann_window(H, W, device)
            gray = gray * win

        if self.freq_mode == "fft":
            # rfft2 returns [B,1,H,(W//2+1)] complex; zero-pad to [B,1,H,W]
            spec_c    = torch.fft.rfft2(gray, norm="ortho")
            magnitude = spec_c.abs()
            # Shift zero-frequency to centre for the CNN to see symmetric patterns
            magnitude = torch.fft.fftshift(magnitude, dim=(-2,))
            F_ = magnitude.shape[-1]
            if F_ < W:
                magnitude = F.pad(magnitude, (0, W - F_))

        else:  # dct
            D    = self._build_dct_matrix(H, device)
            Dw   = self._build_dct_matrix(W, device) if W != H else D
            g    = gray.squeeze(1)                        # [B,H,W]
            # Separable 2-D DCT-II: row transform then column transform
            dct_rows  = torch.einsum("kh,bhw->bkw", D,  g)
            magnitude = torch.einsum("lw,bhw->bhl", Dw, dct_rows).abs().unsqueeze(1)

        log_spec = torch.log1p(magnitude)

        # Per-image min-max normalisation → [0,1]
        flat = log_spec.view(B, -1)
        mn   = flat.min(dim=1).values.view(B, 1, 1, 1)
        mx   = flat.max(dim=1).values.view(B, 1, 1, 1)
        return (log_spec - mn) / (mx - mn + 1e-8)

    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B,3,H,W] in [0,1] → [B, out_dim]"""
        # Explicit luminance weights (ITU-R BT.601)
        gray  = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        spec  = self._compute_spectrum(gray)   # [B,1,H,W]
        feats = self.cnn(spec).flatten(1)      # [B,256]
        return self.head(feats)                # [B,out_dim]


# ---------------------------------------------------------------------------
# Fused Detector
# ---------------------------------------------------------------------------

class FusedDetector(nn.Module):
    """
    Fuses spatial and frequency branches for binary AI-image detection.

    ablation_mode
    -------------
    'fused'        : both branches; output = concat(spatial_proj, freq_proj)
    'spatial_only' : spatial branch only
    'freq_only'    : frequency branch only

    Fusion head: Linear → BN → ReLU → Dropout(0.4) → Linear → 1 logit
    Loss: BCEWithLogitsLoss (raw logit output).

    Optional projection layers (use_projection=True, default):
        Each branch is projected to projection_dim via Linear→BN→ReLU before
        concatenation.  This balances branch capacities and allows the head to
        operate on a fixed-size input regardless of backbone choice.
    """

    def __init__(
        self,
        ablation_mode:  str  = "fused",
        backbone:       str  = "resnet18",
        freq_mode:      str  = "fft",
        use_hann:       bool = True,
        use_projection: bool = True,
        projection_dim: int  = 256,
        freq_cnn_dim:   int  = 256,
    ) -> None:
        super().__init__()
        if ablation_mode not in ("fused", "spatial_only", "freq_only"):
            raise ValueError(
                "ablation_mode must be 'fused', 'spatial_only', or 'freq_only'."
            )
        self.ablation_mode  = ablation_mode
        self.use_projection = use_projection

        self.spatial_branch = SpatialBranch(backbone=backbone, pretrained=True)
        self.freq_branch    = FrequencyBranch(
            freq_mode=freq_mode, use_hann=use_hann, out_dim=freq_cnn_dim
        )

        spatial_dim = self.spatial_branch.out_dim
        freq_dim    = freq_cnn_dim

        if use_projection:
            self.spatial_proj = nn.Sequential(
                nn.Linear(spatial_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
                nn.ReLU(inplace=True),
            )
            self.freq_proj = nn.Sequential(
                nn.Linear(freq_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
                nn.ReLU(inplace=True),
            )
            spatial_out = projection_dim
            freq_out    = projection_dim
        else:
            self.spatial_proj = nn.Identity()
            self.freq_proj    = nn.Identity()
            spatial_out = spatial_dim
            freq_out    = freq_dim

        if ablation_mode == "fused":
            head_in = spatial_out + freq_out
        elif ablation_mode == "spatial_only":
            head_in = spatial_out
        else:
            head_in = freq_out

        hidden = max(head_in // 2, 64)
        self.fusion_head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B,3,H,W] in [0,1] → [B,1] raw logit"""
        sp = fq = None
        if self.ablation_mode in ("fused", "spatial_only"):
            sp = self.spatial_proj(self.spatial_branch(x))
        if self.ablation_mode in ("fused", "freq_only"):
            fq = self.freq_proj(self.freq_branch(x))

        if self.ablation_mode == "fused":
            combined = torch.cat([sp, fq], dim=1)
        elif self.ablation_mode == "spatial_only":
            combined = sp
        else:
            combined = fq

        return self.fusion_head(combined)

    def count_parameters(self) -> Dict[str, int]:
        """Return dict: total, spatial_branch, freq_branch, fusion_head."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        proj_params = (
            _count(self.spatial_proj) + _count(self.freq_proj)
            if self.use_projection else 0
        )
        return {
            "total":          _count(self),
            "spatial_branch": _count(self.spatial_branch),
            "freq_branch":    _count(self.freq_branch),
            "fusion_head":    _count(self.fusion_head) + proj_params,
        }


# ---------------------------------------------------------------------------
# Shape / sanity test
# ---------------------------------------------------------------------------

def test_model_shapes() -> None:
    """
    For each ablation mode, runs a forward pass at both 224×224 and 32×32
    and verifies output shape (B,1). Prints parameter counts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes  = ["fused", "spatial_only", "freq_only"]
    sizes  = [(1, 3, 224, 224), (1, 3, 32, 32)]

    print(f"\n{'='*60}\nModel shape test  |  device: {device}\n{'='*60}")

    for mode in modes:
        model = FusedDetector(ablation_mode=mode).to(device).eval()
        params = model.count_parameters()
        print(f"\n--- ablation_mode='{mode}' ---")
        for k, v in params.items():
            print(f"  {k:<20}: {v:>10,} parameters")

        for shape in sizes:
            dummy = torch.rand(*shape, device=device)
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (shape[0], 1), \
                f"Expected ({shape[0]},1), got {out.shape}"
            print(f"  Input {str(list(shape[1:])):>18}  →  output {list(out.shape)}")

    print(f"\n{'='*60}\n✅ All shape assertions passed.\n{'='*60}\n")


test_model_shapes()

# ===CELL_BREAK===

# [MARKDOWN]
## Cell 10 — DCT Unit Test

# ===CELL_BREAK===

# [CODE]
"""
Cell 10 — DCT Implementation Unit Tests
"""


def test_dct_implementation() -> None:
    """
    Unit test: compare custom GPU DCT-II against scipy.fft.dctn(norm='ortho').
    Uses random [4, 1, 8, 8] tensor; atol=1e-4. Prints PASS or FAIL + max err.
    """
    try:
        from scipy.fft import dctn
    except ImportError:
        print("⚠️  scipy not available — skipping DCT unit test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    B, N = 4, 8

    x_torch = torch.rand(B, 1, N, N, device=device)
    branch  = FrequencyBranch(freq_mode="dct", use_hann=False).to(device).eval()
    D       = branch._build_dct_matrix(N, device)

    g         = x_torch.squeeze(1)
    dct_rows  = torch.einsum("kh,bhw->bkw", D, g)
    our_dct   = torch.einsum("lw,bhw->bhl", D, dct_rows)

    x_np    = x_torch.squeeze(1).cpu().numpy()
    ref_dct = np.stack([dctn(x_np[i], norm="ortho") for i in range(B)])

    our_np  = our_dct.detach().cpu().numpy()
    max_err = float(np.abs(our_np - ref_dct).max())
    atol    = 1e-4

    if max_err <= atol:
        print(f"✅ test_dct_implementation  PASS  (max |err| = {max_err:.2e})")
    else:
        print(f"❌ test_dct_implementation  FAIL  (max |err| = {max_err:.2e}, atol={atol})")
        worst = np.unravel_index(np.abs(our_np - ref_dct).argmax(), our_np.shape)
        b, k, l = worst
        print(f"   Worst @ batch={b}, row={k}, col={l}")
        print(f"   Our: {our_np[b,k,l]:.6f}  Ref: {ref_dct[b,k,l]:.6f}")


def test_hann_window() -> None:
    """
    Verify 2-D Hann window: non-negativity, symmetry, outer-product structure,
    and near-zero corners for several spatial sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch = FrequencyBranch(freq_mode="fft", use_hann=True).to(device).eval()

    for H, W in [(32, 32), (64, 128), (224, 224)]:
        win2d = branch._build_hann_window(H, W, device).squeeze()
        h1d   = torch.hann_window(H, periodic=False, device=device)
        w1d   = torch.hann_window(W, periodic=False, device=device)

        assert win2d.min() >= 0.0, \
            f"[{H}×{W}] Negative values in window!"
        assert torch.allclose(win2d, win2d.flip(0).flip(1), atol=1e-6), \
            f"[{H}×{W}] Window not symmetric!"
        expected = torch.outer(h1d, w1d)
        assert torch.allclose(win2d, expected, atol=1e-6), \
            f"[{H}×{W}] Window != outer product of 1-D Hann windows!"
        for c in [win2d[0, 0], win2d[0, -1], win2d[-1, 0], win2d[-1, -1]]:
            assert c.abs() < 1e-6, \
                f"[{H}×{W}] Corner not near zero: {c:.6f}"

        print(f"✅ test_hann_window [{H}×{W}]  sum={win2d.sum():.2f}  PASS")


test_dct_implementation()
test_hann_window()

# ===CELL_BREAK===

# [MARKDOWN]
## Cell 11 — Trainer

# ===CELL_BREAK===

# [CODE]
"""
Cell 11 — Trainer
"""

import shutil
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from torch.amp import GradScaler as _AmpGradScaler, autocast as _amp_autocast
    def GradScaler():
        return _AmpGradScaler("cuda")
    def autocast():
        return _amp_autocast("cuda")
    _AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler, autocast
        _AMP_AVAILABLE = True
    except ImportError:
        _AMP_AVAILABLE = False

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = lambda x, **kw: x


class Trainer:
    """
    Full training orchestrator for FusedDetector.

    Features
    --------
    - Mixed-precision (AMP) training with gradient clipping.
    - AdamW + cosine-annealing or ReduceLROnPlateau scheduler.
    - Early stopping on val AUC (threshold-free; threshold is set post-hoc).
    - Best-checkpoint persistence with full state dict (model, optimizer,
      scheduler, AMP scaler, cfg, epoch, best_metric, seed).
    - Training-curve plotting (loss | accuracy+AUC panels).
    - Youden's J threshold computed on the VALIDATION set only; never on test.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        cfg:            Any,
        experiment_dir: str,
        seed:           int,
        num_epochs:     Optional[int] = None,
    ) -> None:
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.cfg            = cfg
        self.experiment_dir = Path(experiment_dir)
        self.seed           = seed

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        lr = getattr(cfg, "lr", 1e-4)
        wd = getattr(cfg, "weight_decay", 1e-2)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd,
        )

        if num_epochs is not None:
            n_epochs = num_epochs
        elif getattr(cfg, "QUICK_TEST", False):
            n_epochs = getattr(cfg, "quick_test_epochs", 1)
        else:
            n_epochs = getattr(cfg, "epochs", 20)
        self.n_epochs = n_epochs

        scheduler_name = getattr(cfg, "scheduler", "cosine").lower()
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6,
            )
            self._sched_step_on = "epoch"
        elif scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=max(1, getattr(cfg, "patience", 5) // 2),
                min_lr=1e-6,
            )
            self._sched_step_on = "val_auc"
        else:
            raise ValueError(f"Unknown scheduler '{scheduler_name}'.")

        self.use_amp = _AMP_AVAILABLE and self.device.type == "cuda"
        self.scaler  = GradScaler() if self.use_amp else None

        self.patience     = getattr(cfg, "patience", 5)
        self.best_auc     = -float("inf")
        self.patience_ctr = 0
        self.best_ckpt    = self.experiment_dir / "best_model.pth"

        self.grad_clip = getattr(cfg, "grad_clip", 1.0)

    # -----------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """One training epoch with AMP, gradient clipping, tqdm progress bar."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = _tqdm(self.train_loader, desc="  Train", leave=False, dynamic_ncols=True)
        for batch_x, batch_y in pbar:
            valid = batch_y != -1
            if not valid.any():
                continue
            batch_x = batch_x[valid].to(self.device, non_blocking=True)
            batch_y = batch_y[valid].float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    logits = self.model(batch_x).squeeze(1)
                    loss   = self.criterion(logits, batch_y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch_x).squeeze(1)
                loss   = self.criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            bs          = batch_y.size(0)
            total_loss += loss.item() * bs
            preds       = (torch.sigmoid(logits) > 0.5).long()
            correct    += (preds == batch_y.long()).sum().item()
            total      += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss":     total_loss / max(total, 1),
            "accuracy": correct   / max(total, 1),
        }

    # -----------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate on loader.
        Returns: loss, accuracy, auc, pr_auc, all_probs, all_labels.
        """
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs:  List[float] = []
        all_labels: List[int]   = []

        for batch_x, batch_y in loader:
            valid = batch_y != -1
            if not valid.any():
                continue
            batch_x = batch_x[valid].to(self.device, non_blocking=True)
            batch_y = batch_y[valid].float().to(self.device, non_blocking=True)

            if self.use_amp:
                with autocast():
                    logits = self.model(batch_x).squeeze(1)
                    loss   = self.criterion(logits, batch_y)
            else:
                logits = self.model(batch_x).squeeze(1)
                loss   = self.criterion(logits, batch_y)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()

            bs          = batch_y.size(0)
            total_loss += loss.item() * bs
            correct    += (preds == batch_y.long()).sum().item()
            total      += bs
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch_y.long().cpu().numpy().tolist())

        p_arr    = np.array(all_probs)
        l_arr    = np.array(all_labels)
        n_unique = len(np.unique(l_arr))
        auc      = roc_auc_score(l_arr, p_arr) if n_unique > 1 else 0.5
        pr_auc   = average_precision_score(l_arr, p_arr) if n_unique > 1 else 0.5

        return {
            "loss":       total_loss / max(total, 1),
            "accuracy":   correct   / max(total, 1),
            "auc":        auc,
            "pr_auc":     pr_auc,
            "all_probs":  all_probs,
            "all_labels": all_labels,
        }

    # -----------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and checkpoint saving.

        Checkpoint contains: model_state, optimizer_state, scheduler_state,
        scaler_state, cfg.__dict__, epoch, best_metric (val_auc), seed.

        Returns history dict: train_loss, train_acc, val_loss, val_acc, val_auc.
        """
        n_epochs = self.n_epochs
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_auc": [],
        }

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr = self.train_epoch()
            va = self.evaluate(self.val_loader)
            epoch_time = time.time() - t0
            lr_now     = self.optimizer.param_groups[0]["lr"]

            if self._sched_step_on == "epoch":
                self.scheduler.step()
            else:
                self.scheduler.step(va["auc"])

            print(
                f"Epoch {epoch:03d}/{n_epochs} | lr={lr_now:.2e} | "
                f"train loss={tr['loss']:.4f} acc={tr['accuracy']:.4f} | "
                f"val loss={va['loss']:.4f} acc={va['accuracy']:.4f} "
                f"auc={va['auc']:.4f} pr_auc={va['pr_auc']:.4f} | "
                f"time={epoch_time:.1f}s"
            )

            history["train_loss"].append(tr["loss"])
            history["train_acc"].append(tr["accuracy"])
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["accuracy"])
            history["val_auc"].append(va["auc"])

            if va["auc"] > self.best_auc:
                self.best_auc     = va["auc"]
                self.patience_ctr = 0
                torch.save({
                    "model_state":     self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "scaler_state":    self.scaler.state_dict() if self.scaler else None,
                    "cfg":             vars(self.cfg) if not hasattr(self.cfg, "__dataclass_fields__")
                                       else {f.name: getattr(self.cfg, f.name)
                                             for f in self.cfg.__dataclass_fields__.values()},
                    "epoch":           epoch,
                    "best_metric":     self.best_auc,
                    "seed":            self.seed,
                }, self.best_ckpt)
                print(f"  💾 Saved best checkpoint (val_auc={self.best_auc:.4f})")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.patience:
                    print(
                        f"  ⏹  Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)."
                    )
                    break

        return history

    # -----------------------------------------------------------------------

    def plot_curves(
        self, history: Dict[str, List[float]], experiment_dir: str
    ) -> None:
        """
        2-panel figure: (1) train/val loss, (2) train/val accuracy + val AUC.
        Saves to experiment_dir/training_curves.png and ROOT/plots/.
        """
        exp_dir    = Path(experiment_dir)
        root_plots = Path(getattr(self.cfg, "ROOT", ".")) / "plots"
        root_plots.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", ms=4)
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", ms=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("Loss Curves")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(epochs, history["train_acc"], label="Train Acc",  marker="o", ms=4)
        ax.plot(epochs, history["val_acc"],   label="Val Acc",    marker="s", ms=4)
        ax.plot(epochs, history["val_auc"],   label="Val AUC",    marker="^", ms=4,
                linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Accuracy & AUC")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

        fig.suptitle("Training Curves", fontsize=14, y=1.02)
        fig.tight_layout()

        out_path = exp_dir / "training_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        try:
            shutil.copy2(out_path, root_plots / out_path.name)
        except Exception:
            pass
        plt.close(fig)
        print(f"  📈 Curves saved → {out_path}")

    # -----------------------------------------------------------------------

    def compute_youden_threshold(
        self, val_probs: List[float], val_labels: List[int]
    ) -> float:
        """
        Youden's J = Sensitivity + Specificity − 1.
        Computed on the VALIDATION set only.
        NEVER call this with test data — that would constitute threshold leakage.
        """
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
        j_scores             = tpr + (1 - fpr) - 1.0
        best_idx             = int(np.argmax(j_scores))
        thr                  = float(np.clip(thresholds[best_idx], 0.0, 1.0))
        print(
            f"  Youden threshold (val): {thr:.4f}  "
            f"(J={j_scores[best_idx]:.4f}, TPR={tpr[best_idx]:.4f}, "
            f"Specificity={1 - fpr[best_idx]:.4f})"
        )
        return thr


print("✅ Cell 11: Trainer class defined.")

# ===CELL_BREAK===

# [MARKDOWN]
## Cell 12 — Full Test Evaluation

# ===CELL_BREAK===

# [CODE]
"""
Cell 12 — Full Test Evaluation: metrics, plots, latency benchmarking.
"""

import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    balanced_accuracy_score,
)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_best_checkpoint(
    experiment_dir: str,
    model: nn.Module,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load best_model.pth from experiment_dir.
    Returns (model in eval mode with weights loaded, full checkpoint dict).
    """
    ckpt_path = Path(experiment_dir) / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    print(
        f"✅ Loaded checkpoint  epoch={checkpoint.get('epoch', '?')}  "
        f"val_auc={checkpoint.get('best_metric', float('nan')):.4f}"
    )
    return model, checkpoint


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def _benchmark_latency(
    model:       nn.Module,
    device:      torch.device,
    batch_sizes: Tuple[int, ...] = (1, 64),
    image_size:  int = 224,
    n_warmup:    int = 10,
    n_runs:      int = 50,
) -> Dict[str, float]:
    """
    Measure per-image inference latency at given batch sizes after GPU warm-up.

    Returns dict: {'latency_bs1_ms': ..., 'latency_bs64_ms': ...}
    where the value is ms per *image* (not per batch).
    """
    model.eval()
    results: Dict[str, float] = {}
    with torch.no_grad():
        for bs in batch_sizes:
            dummy = torch.rand(bs, 3, image_size, image_size, device=device)
            # GPU warm-up
            for _ in range(n_warmup):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ms_per_img = elapsed_ms / (n_runs * bs)
            key = f"latency_bs{bs}_ms"
            results[key] = round(ms_per_img, 4)
            print(f"  Latency (bs={bs:>3}): {ms_per_img:.4f} ms/image")
    return results


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model:          nn.Module,
    test_loader:    DataLoader,
    val_probs:      List[float],
    val_labels:     List[int],
    experiment_dir: str,
    cfg:            Any,
    seed:           int,
) -> Dict[str, Any]:
    """
    Full test-set evaluation: primary (threshold=0.5) and secondary
    (Youden's J threshold fixed from val set) metrics, publication-quality
    plots, latency benchmarking, and JSON export.

    Parameters
    ----------
    model          : trained FusedDetector (eval mode).
    test_loader    : test-set DataLoader.
    val_probs      : predicted probabilities on the VALIDATION set.
    val_labels     : ground-truth labels on the VALIDATION set.
    experiment_dir : output directory for all artefacts.
    cfg            : global config object.
    seed           : experiment seed (logged in output JSON).

    Threshold protocol
    ------------------
    Youden's J is computed on the VALIDATION set.  The same threshold is then
    applied unchanged to the TEST set.  Test data is never used to select or
    tune the threshold.

    Returns
    -------
    full_metrics : dict with all metrics, paths, and latency figures.
    """
    exp_dir    = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    root_plots = Path(getattr(cfg, "ROOT", ".")) / "plots"
    root_plots.mkdir(parents=True, exist_ok=True)

    device  = next(model.parameters()).device
    use_amp = _AMP_AVAILABLE and device.type == "cuda"

    # -----------------------------------------------------------------------
    # 1. Collect test predictions
    # -----------------------------------------------------------------------
    model.eval()
    all_probs:  List[float] = []
    all_labels: List[int]   = []

    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y in _tqdm(
            test_loader, desc="  Evaluating test set", leave=False
        ):
            valid = batch_y != -1
            if not valid.any():
                continue
            batch_x = batch_x[valid].to(device, non_blocking=True)
            batch_y = batch_y[valid]
            if use_amp:
                with autocast():
                    logits = model(batch_x).squeeze(1)
            else:
                logits = model(batch_x).squeeze(1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(batch_y.numpy().tolist())
    t_inf_end = time.perf_counter()

    all_probs_np  = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    n_test        = len(all_labels_np)
    total_inf_ms  = (t_inf_end - t_inf_start) * 1000.0

    # -----------------------------------------------------------------------
    # 2. Youden threshold — computed from VALIDATION set, NOT test set
    # -----------------------------------------------------------------------
    val_probs_np  = np.array(val_probs)
    val_labels_np = np.array(val_labels)
    fpr_v, tpr_v, thr_v = roc_curve(val_labels_np, val_probs_np)
    j_scores             = tpr_v + (1 - fpr_v) - 1.0
    best_j_idx           = int(np.argmax(j_scores))
    youden_threshold     = float(np.clip(thr_v[best_j_idx], 0.0, 1.0))
    youden_j_val         = float(j_scores[best_j_idx])
    print(
        f"  Youden threshold (from val): {youden_threshold:.4f}  "
        f"(J_val={youden_j_val:.4f})"
    )

    # -----------------------------------------------------------------------
    # 3. Metric computation helper
    # -----------------------------------------------------------------------
    def _metrics_at(probs: np.ndarray, labels: np.ndarray, thr: float) -> Dict[str, Any]:
        preds = (probs >= thr).astype(int)
        cm_vals = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm_vals.ravel()
        specificity = float(tn / (tn + fp + 1e-10))
        return {
            "threshold":          float(thr),
            "accuracy":           float(accuracy_score(labels, preds)),
            "precision":          float(precision_score(labels, preds, zero_division=0)),
            "recall":             float(recall_score(labels, preds, zero_division=0)),
            "f1":                 float(f1_score(labels, preds, zero_division=0)),
            "specificity":        specificity,
            "balanced_accuracy":  float(balanced_accuracy_score(labels, preds)),
            "roc_auc":            float(roc_auc_score(labels, probs)),
            "pr_auc":             float(average_precision_score(labels, probs)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }

    metrics_05     = _metrics_at(all_probs_np, all_labels_np, 0.5)
    metrics_youden = _metrics_at(all_probs_np, all_labels_np, youden_threshold)
    clf_report_str = classification_report(
        all_labels_np, (all_probs_np >= 0.5).astype(int),
        target_names=["Real", "AI-Generated"],
    )

    # -----------------------------------------------------------------------
    # 4. Parameter counts + latency benchmarks
    # -----------------------------------------------------------------------
    param_counts = (
        model.count_parameters()
        if hasattr(model, "count_parameters")
        else {"total": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    )
    # Use native crop size or common_crop for latency benchmark input
    lat_img_size = getattr(cfg, "common_crop", 32)
    latency_dict = _benchmark_latency(model, device, batch_sizes=(1, 64),
                                      image_size=lat_img_size,
                                      n_warmup=10, n_runs=50)

    # -----------------------------------------------------------------------
    # 5. Confusion matrix heatmap (seaborn)
    # -----------------------------------------------------------------------
    cm_arr = confusion_matrix(
        all_labels_np, (all_probs_np >= 0.5).astype(int), labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real", "AI-Generated"],
        yticklabels=["Real", "AI-Generated"],
        ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (threshold = 0.5)")
    fig.tight_layout()
    cm_path = exp_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    try:
        shutil.copy2(cm_path, root_plots / cm_path.name)
    except Exception:
        pass
    plt.close(fig)
    print(f"  🗂  Confusion matrix → {cm_path}")

    # -----------------------------------------------------------------------
    # 6. ROC curve
    # -----------------------------------------------------------------------
    fpr_t, tpr_t, _ = roc_curve(all_labels_np, all_probs_np)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_t, tpr_t, lw=2, label=f"AUC = {metrics_05['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.scatter(
        [1 - metrics_youden["specificity"]],
        [metrics_youden["recall"]],
        color="red", zorder=5, s=80,
        label=f"Youden (thr={youden_threshold:.2f}, J={youden_j_val:.3f})",
    )
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    roc_path = exp_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    try:
        shutil.copy2(roc_path, root_plots / roc_path.name)
    except Exception:
        pass
    plt.close(fig)
    print(f"  📉 ROC curve → {roc_path}")

    # -----------------------------------------------------------------------
    # 7. Precision-Recall curve
    # -----------------------------------------------------------------------
    prec_arr, rec_arr, _ = precision_recall_curve(all_labels_np, all_probs_np)
    baseline = float(all_labels_np.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, lw=2, label=f"PR-AUC = {metrics_05['pr_auc']:.4f}")
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline prevalence = {baseline:.2f}")
    ax.scatter(
        [metrics_youden["recall"]],
        [metrics_youden["precision"]],
        color="red", zorder=5, s=80,
        label=f"Youden (thr={youden_threshold:.2f})",
    )
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    pr_path = exp_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    try:
        shutil.copy2(pr_path, root_plots / pr_path.name)
    except Exception:
        pass
    plt.close(fig)
    print(f"  📈 PR curve → {pr_path}")

    # -----------------------------------------------------------------------
    # 8. Summary printout
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TEST SET EVALUATION SUMMARY")
    print("=" * 65)
    print(f"\n--- Threshold = 0.5 ---")
    for k, v in metrics_05.items():
        print(f"  {k:<25}: {v}")
    print(f"\n--- Threshold = Youden's J ({youden_threshold:.4f}, computed on val) ---")
    for k, v in metrics_youden.items():
        print(f"  {k:<25}: {v}")
    print(f"\n--- Classification Report (threshold=0.5) ---\n{clf_report_str}")
    print("--- Parameter Counts ---")
    for k, v in param_counts.items():
        print(f"  {k:<25}: {v:,}")
    print("\n--- Latency ---")
    for k, v in latency_dict.items():
        print(f"  {k:<35}: {v:.4f} ms/image")
    bulk_ms = total_inf_ms / max(n_test, 1)
    print(f"  {'bulk_inference_ms_per_img':<35}: {bulk_ms:.4f} ms/image")
    print("=" * 65 + "\n")

    # -----------------------------------------------------------------------
    # 9. Persist metrics to JSON + text
    # -----------------------------------------------------------------------
    full_metrics: Dict[str, Any] = {
        "seed":                      seed,
        "n_test_samples":            n_test,
        "threshold_0_5":             metrics_05,
        "youden_threshold":          youden_threshold,
        "youden_j_on_val":           youden_j_val,
        "threshold_youden":          metrics_youden,
        "classification_report_str": clf_report_str,
        "param_counts":              param_counts,
        "latency":                   latency_dict,
        "bulk_inference_ms_per_img": bulk_ms,
        "experiment_dir":            str(experiment_dir),
        "plots": {
            "confusion_matrix": str(cm_path),
            "roc_curve":        str(roc_path),
            "pr_curve":         str(pr_path),
        },
    }

    # JSON-safe copy (exclude the long string report from JSON, keep it in .txt)
    json_safe = {k: v for k, v in full_metrics.items()
                 if k != "classification_report_str"}

    metrics_json_path = exp_dir / "test_metrics.json"
    with open(metrics_json_path, "w") as fh:
        json.dump(json_safe, fh, indent=2, default=str)

    with open(exp_dir / "classification_report.txt", "w") as fh:
        fh.write(clf_report_str)

    print(f"  💾 Metrics JSON → {metrics_json_path}")
    print(f"  💾 Classification report → {exp_dir / 'classification_report.txt'}")

    return full_metrics


print("✅ Cell 12: evaluate_test_set + load_best_checkpoint + _benchmark_latency defined.")
