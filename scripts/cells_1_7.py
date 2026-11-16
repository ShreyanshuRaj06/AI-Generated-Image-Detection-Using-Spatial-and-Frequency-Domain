# [MARKDOWN]
# AI-Generated Image Detection: Spatial & Frequency Domain Features
## Research Notebook — Google Colab (T4 GPU)
**Central question:** Does fusing spatial-domain and frequency-domain features improve AI-generated image detection, particularly on datasets unseen during training?
- **Task:** Binary classification — Real = 0, AI-generated = 1
- **Environment:** Google Colab T4 GPU, ~12 GB RAM, Google Drive (free 15 GB)
- Run cells top-to-bottom. All hyperparameters live in the `CFG` dataclass (Cell 3).

# ===CELL_BREAK===

# [CODE]
# Install/upgrade all required packages
# Note: scipy needed for DCT unit test, seaborn for plots
!pip install -q --upgrade torch torchvision torchaudio
!pip install -q timm scipy seaborn scikit-learn tqdm Pillow pandas matplotlib

# ===CELL_BREAK===

# [CODE]
from __future__ import annotations

import os
import random
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np


@dataclass
class CFG:
    # ── Paths ──────────────────────────────────────────────────────────────
    ROOT: str = "/content/drive/MyDrive/Image_Forensics_Outputs/"
    SUBSET_ROOT: str = "/content/subsets/"

    # ── Reproducibility ────────────────────────────────────────────────────
    seeds: List[int] = field(default_factory=lambda: [42, 123, 2025])

    # ── Dataset budget ─────────────────────────────────────────────────────
    max_samples_per_class: int = 20_000  # per split before 70/15/15

    # ── Quick-test mode (smoke test without full data) ─────────────────────
    QUICK_TEST: bool = False          # 500 images/class, 1 epoch when True
    quick_test_samples: int = 500
    quick_test_epochs: int = 1

    # ── DataLoader ─────────────────────────────────────────────────────────
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Training schedule ──────────────────────────────────────────────────
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 5             # early stopping on val AUC

    # ── LR scheduler ──────────────────────────────────────────────────────
    scheduler: str = "cosine"     # "cosine" or "plateau"

    # ── Ablation mode ─────────────────────────────────────────────────────
    ablation_mode: str = "fused"  # "fused" | "spatial_only" | "freq_only"

    # ── Model / backbone ──────────────────────────────────────────────────
    backbone: str = "resnet18"    # "resnet18" or "mobilenet_v2"

    # ── Frequency branch ──────────────────────────────────────────────────
    freq_mode: str = "fft"        # "fft" or "dct"
    use_hann: bool = True         # apply Hann window before FFT/DCT
    use_projection: bool = True   # project spatial + freq embeddings
    projection_dim: int = 256     # output dim of projection head
    freq_cnn_dim: int = 256       # intermediate dim of frequency CNN branch

    # ── Near-duplicate detection ───────────────────────────────────────────
    use_phash: bool = False        # optional pHash dedup (slow)

    # ── Storage / Drive management ────────────────────────────────────────
    process_per_dataset: bool = True      # process one dataset at a time
    delete_after_training: bool = False   # delete local subsets after training
    min_drive_free_gb: float = 1.0        # abort if Drive has less than this

    # ── Crop protocol ─────────────────────────────────────────────────────
    common_crop: int = 32         # Protocol C (cross-dataset) crop size
    dataset_crop_sizes: Dict[str, int] = field(default_factory=lambda: {
        "GRAVEX":    224,
        "FACES":     224,
        "CASHBOWMAN":224,
        "CIFAKE":    32,
    })

    # ── Split quality gates ────────────────────────────────────────────────
    val_min_samples: int = 500
    test_min_samples: int = 500

    # ── Dataset registry ──────────────────────────────────────────────────
    datasets: List[str] = field(default_factory=lambda: [
        "GRAVEX", "FACES", "CASHBOWMAN", "CIFAKE"
    ])
    dataset_kaggle_ids: Dict[str, str] = field(default_factory=lambda: {
        "GRAVEX":    "muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal",
        "FACES":     "xhlulu/140k-real-and-fake-faces",
        "CASHBOWMAN":"cashbowman/ai-generated-images-vs-real-images",
        "CIFAKE":    "birdy654/cifake-real-and-ai-generated-synthetic-images",
    })

    # ── Runtime registries (populated by DownloadManager) ──────────────────
    manifests:    Dict[str, str] = field(default_factory=dict)
    subset_roots: Dict[str, str] = field(default_factory=dict)

    # ── ImageNet normalisation ─────────────────────────────────────────────
    imagenet_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    imagenet_std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


# ── Seed utility ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic ops where possible (may slow things down slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_cfg() -> CFG:
    """Return a default CFG instance."""
    return CFG()


# ── Instantiate and display ───────────────────────────────────────────────────

CFG_INSTANCE = get_cfg()
print("=" * 60)
print("CFG (default):")
print("=" * 60)
for f in dataclasses.fields(CFG_INSTANCE):
    print(f"  {f.name:30s} = {getattr(CFG_INSTANCE, f.name)}")
print("=" * 60)

# ===CELL_BREAK===

# [CODE]
import os
import shutil
from pathlib import Path

from google.colab import drive  # type: ignore[import]


# ── Mount Drive ───────────────────────────────────────────────────────────────

drive.mount("/content/drive", force_remount=False)
print("✅ Google Drive mounted at /content/drive")


# ── Space checker ─────────────────────────────────────────────────────────────

def check_drive_space(min_free_gb: float, cfg: CFG) -> None:
    """Check free space on Google Drive; abort if below threshold.

    Uses shutil.disk_usage on the ROOT directory (Drive must already be
    mounted).  Raises RuntimeError with a human-readable message when the
    available space falls below *min_free_gb* gigabytes.
    """
    root = Path(cfg.ROOT)
    root.mkdir(parents=True, exist_ok=True)   # ensure it exists before stat

    usage = shutil.disk_usage(str(root))
    free_gb  = usage.free  / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    used_gb  = usage.used  / (1024 ** 3)

    print(
        f"Drive usage — total: {total_gb:.2f} GB  "
        f"used: {used_gb:.2f} GB  free: {free_gb:.2f} GB"
    )

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient Google Drive space: {free_gb:.2f} GB free, "
            f"but at least {min_free_gb:.2f} GB is required.  "
            "Free up space on Google Drive and re-run this cell."
        )

    print(f"✅ Drive space OK ({free_gb:.2f} GB free ≥ {min_free_gb:.2f} GB required)")


# ── Directory setup ───────────────────────────────────────────────────────────

def setup_directories(cfg: CFG) -> None:
    """Create ROOT and SUBSET_ROOT directory trees."""
    root = Path(cfg.ROOT)
    subdirs = ["manifests", "checkpoints", "results", "plots", "configs"]
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)
        print(f"  📁 {root / sub}")

    Path(cfg.SUBSET_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"  📁 {cfg.SUBSET_ROOT}")
    print("✅ All directories created / verified.")


# ── Run ───────────────────────────────────────────────────────────────────────

cfg = get_cfg()
set_seed(cfg.seeds[0])

check_drive_space(cfg.min_drive_free_gb, cfg)
setup_directories(cfg)

# ===CELL_BREAK===

# [CODE]
import os
import shutil
import stat
from pathlib import Path


def setup_kaggle() -> None:
    """
    Locate kaggle.json, copy it to ~/.kaggle/kaggle.json and set
    permissions to 600 so the Kaggle CLI accepts it.

    Search order
    ------------
    1. /content/kaggle.json   (uploaded via Colab file panel)
    2. /root/kaggle.json
    3. /root/.config/kaggle/kaggle.json
    4. ~/.kaggle/kaggle.json  (already in place — just chmod)
    """
    candidate_paths = [
        Path("/content/kaggle.json"),
        Path("/root/kaggle.json"),
        Path("/root/.config/kaggle/kaggle.json"),
    ]

    dest_dir  = Path.home() / ".kaggle"
    dest_file = dest_dir / "kaggle.json"

    dest_dir.mkdir(parents=True, exist_ok=True)

    source: Path | None = None

    # If the file is already in place, we still re-chmod it.
    if dest_file.exists():
        source = dest_file
        print(f"ℹ️  kaggle.json already present at {dest_file}")
    else:
        for candidate in candidate_paths:
            if candidate.exists():
                source = candidate
                print(f"📄 Found kaggle.json at {source}")
                break

    if source is None:
        raise FileNotFoundError(
            "kaggle.json not found in any of the expected locations:\n"
            + "\n".join(f"  • {p}" for p in [*candidate_paths, dest_file])
            + "\n\nUpload your kaggle.json via the Colab ➜ Files panel "
            "or place it at /content/kaggle.json and re-run this cell."
        )

    if source != dest_file:
        shutil.copy2(str(source), str(dest_file))
        print(f"✅ Copied {source} → {dest_file}")
    else:
        print(f"✅ Using existing {dest_file}")

    # chmod 600
    dest_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"🔒 Permissions set to 600 on {dest_file}")

    # Quick sanity check — the file should contain "username" and "key"
    text = dest_file.read_text()
    if '"username"' not in text or '"key"' not in text:
        raise ValueError(
            f"{dest_file} does not look like a valid kaggle.json "
            "(missing 'username' or 'key').  Please re-download it from "
            "https://www.kaggle.com/settings → 'Create New Token'."
        )

    print("✅ kaggle.json validated.  Kaggle CLI is ready.")


setup_kaggle()

# ===CELL_BREAK===

# [CODE]
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm


# ── 1. Directory-tree inspector ───────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_dataset_tree(
    raw_dir: str,
    max_depth: int = 4,
) -> Dict[str, int]:
    """Walk *raw_dir* up to *max_depth* levels deep, printing a tree with
    per-folder image counts.

    Returns
    -------
    dict
        ``{relative_folder_path: image_count}`` for every visited folder.
        Counts are *direct-child* image files only (not recursive).
    """
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    result: Dict[str, int] = {}

    def _walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            return

        # Count direct-child images
        images = [
            e for e in entries
            if e.is_file() and e.suffix.lower() in _IMAGE_EXTS
        ]
        rel = str(directory.relative_to(root))
        result[rel] = len(images)

        # Print line
        label = directory.name
        count_str = f"  [{len(images)} imgs]" if images else ""
        print(f"{prefix}📂 {label}{count_str}")

        subdirs = [e for e in entries if e.is_dir()]
        for i, sub in enumerate(subdirs):
            connector  = "└── " if i == len(subdirs) - 1 else "├── "
            child_pfx  = prefix + ("    " if i == len(subdirs) - 1 else "│   ")
            # Inline sub-entry count on the connector line
            sub_imgs = [
                f for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
            ] if sub.is_dir() else []
            sub_count_str = f"  [{len(sub_imgs)} imgs]" if sub_imgs else ""
            print(f"{prefix}{connector}📂 {sub.name}{sub_count_str}")
            _walk(sub, child_pfx, depth + 1)

    print(f"\n🗂️  Dataset tree: {raw_dir}")
    print("=" * 60)
    _walk(root, "", 1)
    print("=" * 60)
    total = sum(result.values())
    print(f"Total image files found (direct per folder): {total}\n")
    return result


# ── 2. Sample grid visualiser ─────────────────────────────────────────────────

def show_sample_grid(
    paths: List[str],
    labels: List[int],
    label_map_inv: Dict[int, str],
    title: str,
    n: int = 4,
    cfg: "CFG | None" = None,
) -> None:
    """Display an *n*×2 grid: *n* real samples (label=0) then *n* fake
    samples (label=1).  Saves the figure to ROOT/plots/ when *cfg* is
    provided.

    Parameters
    ----------
    paths         : flat list of file paths
    labels        : corresponding integer labels (0 = real, 1 = fake)
    label_map_inv : ``{0: 'real', 1: 'fake'}`` (or similar)
    title         : figure super-title
    n             : columns per class row  (shows up to *n* of each class)
    cfg           : CFG instance; if None the figure is not saved to Drive.
    """
    real_paths = [p for p, l in zip(paths, labels) if l == 0]
    fake_paths = [p for p, l in zip(paths, labels) if l == 1]

    real_sample = real_paths[:n]
    fake_sample = fake_paths[:n]

    n_real = len(real_sample)
    n_fake = len(fake_sample)

    if n_real == 0 and n_fake == 0:
        print("⚠️  No images to display.")
        return

    cols = max(n_real, n_fake, 1)
    rows = 2  # row 0 = real, row 1 = fake

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 1))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Ensure axes is always 2-D
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    real_label_str = label_map_inv.get(0, "Real")
    fake_label_str = label_map_inv.get(1, "Fake")

    for row_idx, (sample, row_label) in enumerate(
        [(real_sample, real_label_str), (fake_sample, fake_label_str)]
    ):
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(sample):
                try:
                    img = Image.open(sample[col_idx]).convert("RGB")
                    ax.imshow(np.asarray(img))
                    if col_idx == 0:
                        ax.set_ylabel(row_label, fontsize=11, rotation=90, labelpad=4)
                except Exception as exc:
                    ax.text(0.5, 0.5, f"ERR\n{exc}", ha="center", va="center",
                            fontsize=7, transform=ax.transAxes)

    plt.tight_layout()

    if cfg is not None:
        plots_dir = Path(cfg.ROOT) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
        save_path  = plots_dir / f"{safe_title}.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"💾 Sample grid saved to {save_path}")

    plt.show()
    plt.close(fig)


# ── 3. SHA-256 hasher ─────────────────────────────────────────────────────────

def compute_sha256(filepath: str, chunk_size: int = 65_536) -> str:
    """Return the hex-encoded SHA-256 digest of *filepath* using
    chunked reads to keep memory usage low for large files."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        raise OSError(f"Cannot read file for hashing: {filepath}") from exc


# ── 4. Image metadata extractor ───────────────────────────────────────────────

def get_image_info(filepath: str) -> Tuple[int, int, str]:
    """Return (width, height, format_string) for *filepath*.

    Format is normalised to "JPEG", "PNG", or the PIL-reported format
    string.  Returns ``(-1, -1, "ERROR")`` on any failure.
    """
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            fmt = (img.format or Path(filepath).suffix.lstrip(".").upper())
            # Normalise common aliases
            fmt = {"JPG": "JPEG"}.get(fmt.upper(), fmt.upper())
            return w, h, fmt
    except (UnidentifiedImageError, OSError, Exception):
        return -1, -1, "ERROR"


print("✅ Utility functions defined: inspect_dataset_tree, show_sample_grid, "
      "compute_sha256, get_image_info")

# ===CELL_BREAK===

# [CODE]
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


class DownloadManager:
    """End-to-end pipeline manager: download → inspect → label → hash →
    split → dedup → crop → manifest for one dataset at a time."""

    # Keyword sets used for label inference (lower-cased)
    _REAL_KEYWORDS  = {"real", "real_images", "genuine", "authentic", "original"}
    _FAKE_KEYWORDS  = {"fake", "ai", "ai_generated", "synthetic", "generated",
                       "artificial", "forge", "forged"}

    def __init__(self, cfg: "CFG") -> None:
        self.cfg = cfg

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Download
    # ─────────────────────────────────────────────────────────────────────────

    def download_dataset(self, dataset_name: str) -> str:
        """Download and unzip one dataset via kaggle CLI.

        Returns the path to the extracted raw directory.
        """
        kaggle_id = self.cfg.dataset_kaggle_ids.get(dataset_name)
        if kaggle_id is None:
            raise KeyError(f"No Kaggle ID registered for dataset '{dataset_name}'")

        raw_dir = Path(f"/content/raw_{dataset_name.lower()}")
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"⬇️  Downloading {dataset_name} ({kaggle_id}) …")
        print(f"{'='*60}")

        kaggle_bin = shutil.which("kaggle")
        if kaggle_bin:
            cmd = [
                kaggle_bin, "datasets", "download",
                "-d", kaggle_id,
                "-p", str(raw_dir),
                "--unzip",
            ]
        else:
            cmd = [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", kaggle_id,
                "-p", str(raw_dir),
                "--unzip",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and kaggle_bin:
            # Fallback to python -m kaggle if standalone kaggle binary failed
            cmd_fallback = [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", kaggle_id,
                "-p", str(raw_dir),
                "--unzip",
            ]
            res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res_fb.returncode == 0:
                result = res_fb

        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed for {dataset_name} ({kaggle_id}):\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\n"
                "Ensure kaggle.json is configured (Cell 5) and credentials are valid."
            )

        print(result.stdout.strip())
        print(f"✅ {dataset_name} downloaded to {raw_dir}")
        return str(raw_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Label map
    # ─────────────────────────────────────────────────────────────────────────

    def build_label_map(
        self, dataset_name: str, raw_dir: str
    ) -> Tuple[List[Tuple[str, int, Optional[str]]], Dict[str, int]]:
        """Inspect tree, infer label map from folder names, return file list.

        Returns
        -------
        file_list : list of (filepath, label, split_hint)
        label_map : {folder_name_lower: label_int}
        """
        print(f"\n🔍 Inspecting {dataset_name} …")
        inspect_dataset_tree(raw_dir, max_depth=5)

        root = Path(raw_dir)
        split_keywords = {"train", "training", "val", "valid", "validation",
                          "test", "testing", "eval"}

        # Collect all leaf-ish folders that actually contain images
        image_folders: List[Tuple[Path, Optional[str]]] = []  # (folder, split_hint)

        for dirpath, dirnames, filenames in os.walk(raw_dir):
            imgs = [
                f for f in filenames
                if Path(f).suffix.lower() in _IMAGE_EXTS
            ]
            if not imgs:
                continue
            folder = Path(dirpath)
            # Determine split hint from ancestry
            parts_lower = [p.lower() for p in folder.relative_to(root).parts]
            split_hint: Optional[str] = None
            for part in parts_lower:
                if part in {"train", "training"}:
                    split_hint = "train"; break
                if part in {"val", "valid", "validation"}:
                    split_hint = "val"; break
                if part in {"test", "testing", "eval"}:
                    split_hint = "test"; break
            image_folders.append((folder, split_hint))

        if not image_folders:
            raise ValueError(f"No image-containing folders found under {raw_dir}")

        # Build label map from the *immediate parent* folder of each image
        # group, excluding generic split-keyword folders as the label source.
        all_label_folder_names: Dict[str, int] = {}
        ambiguous: List[str] = []

        def _infer_label(name: str) -> int | None:
            nl = name.lower().strip()
            is_real = nl in self._REAL_KEYWORDS or any(k in nl for k in self._REAL_KEYWORDS)
            is_fake = nl in self._FAKE_KEYWORDS or any(k in nl for k in self._FAKE_KEYWORDS)
            if is_real and not is_fake:
                return 0
            if is_fake and not is_real:
                return 1
            return None

        label_map: Dict[str, int] = {}

        for folder, _split in image_folders:
            # Walk up the path (excluding root) to find a label-bearing folder
            rel_parts = list(folder.relative_to(root).parts)
            label_found = False
            for part in reversed(rel_parts):
                if part.lower() in split_keywords:
                    continue          # skip train/val/test names
                label = _infer_label(part)
                if label is not None:
                    label_map[part] = label
                    label_found = True
                    break
                # Ambiguous — flag it
                ambiguous.append(part)
                break

        if not label_map:
            raise ValueError(
                f"Could not infer any label mapping for {dataset_name}.\n"
                f"Folders found: {[str(f) for f, _ in image_folders]}\n"
                "Please add new keywords to _REAL_KEYWORDS / _FAKE_KEYWORDS."
            )

        ambiguous_filtered = [a for a in ambiguous
                              if a not in label_map and a.lower() not in split_keywords]
        if ambiguous_filtered:
            raise ValueError(
                f"Ambiguous folder name(s) in {dataset_name}: {ambiguous_filtered}\n"
                f"Could not assign label 0/1.  Label map so far: {label_map}"
            )

        print(f"\n📋 Label map for {dataset_name}:")
        for name, lbl in sorted(label_map.items()):
            print(f"   '{name}'  →  {lbl} ({'real' if lbl == 0 else 'AI-generated'})")

        # Build the file list
        file_list: List[Tuple[str, int, Optional[str]]] = []

        for folder, split_hint in image_folders:
            rel_parts = list(folder.relative_to(root).parts)
            # Determine label for this folder
            label: Optional[int] = None
            for part in reversed(rel_parts):
                if part.lower() in split_keywords:
                    continue
                label = label_map.get(part)
                if label is not None:
                    break

            if label is None:
                warnings.warn(
                    f"Skipping folder with unknown label: {folder}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for fname in os.listdir(str(folder)):
                if Path(fname).suffix.lower() not in _IMAGE_EXTS:
                    continue
                fpath = str(folder / fname)
                file_list.append((fpath, label, split_hint))

        print(f"\n📦 Total files collected: {len(file_list)}")
        label_counts = {}
        for _, lbl, _ in file_list:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        for lbl, cnt in sorted(label_counts.items()):
            print(f"   Label {lbl}: {cnt} files")

        return file_list, label_map

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Hash + metadata
    # ─────────────────────────────────────────────────────────────────────────

    def compute_hashes_and_info(
        self, file_list: List[Tuple[str, int, Optional[str]]]
    ) -> pd.DataFrame:
        """Hash every file and collect width/height/format metadata."""
        print(f"\n🔐 Computing SHA-256 hashes and image metadata …")
        records = []
        for fpath, label, split_hint in tqdm(file_list, desc="Hashing", unit="img"):
            try:
                sha = compute_sha256(fpath)
            except OSError:
                sha = "ERROR"
            w, h, fmt = get_image_info(fpath)
            records.append({
                "filepath":    fpath,
                "label":       label,
                "split_hint":  split_hint,
                "sha256":      sha,
                "orig_width":  w,
                "orig_height": h,
                "orig_format": fmt,
            })
        df = pd.DataFrame(records)
        print(f"✅ Hashed {len(df)} files.  "
              f"Errors: {(df['sha256'] == 'ERROR').sum()}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Split assignment
    # ─────────────────────────────────────────────────────────────────────────

    def assign_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preserve shipped splits or create a stratified 70/15/15 split."""
        cfg = self.cfg
        seed = cfg.seeds[0]

        df = df.copy()
        has_shipped = df["split_hint"].notna() & df["split_hint"].isin(
            ["train", "val", "test"]
        )
        shipped_ratio = has_shipped.mean()

        if shipped_ratio > 0.5:
            print(f"ℹ️  {shipped_ratio:.0%} of files have a shipped split — preserving them.")
            df["split"] = df["split_hint"].where(has_shipped, other=None)
            # Files without a split hint get assigned to train
            no_hint = df["split"].isna()
            if no_hint.any():
                df.loc[no_hint, "split"] = "train"
                print(f"   {no_hint.sum()} files with no split hint → 'train'")
        else:
            print("ℹ️  No shipped splits detected — creating stratified 70/15/15 split.")
            df["split"] = None
            idx = df.index.tolist()
            labels = df["label"].tolist()

            train_idx, temp_idx, _, temp_labels = train_test_split(
                idx, labels, test_size=0.30, stratify=labels, random_state=seed
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed
            )

            df.loc[train_idx, "split"] = "train"
            df.loc[val_idx,   "split"] = "val"
            df.loc[test_idx,  "split"] = "test"

        for split_name in ["train", "val", "test"]:
            cnt = (df["split"] == split_name).sum()
            print(f"   {split_name:5s}: {cnt:>6d} files")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Deduplication
    # ─────────────────────────────────────────────────────────────────────────

    def deduplicate(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Remove exact duplicates within and across splits."""
        df = df.copy()
        report: Dict = {}
        initial = len(df)

        # --- Step 1: intra-train dedup (keep first occurrence by sha256)
        train_mask = df["split"] == "train"
        train_df   = df[train_mask]
        duped_train = train_df.duplicated(subset=["sha256"], keep="first")
        dropped_train = int(duped_train.sum())
        df = df[~(train_mask & duped_train)]
        report["intra_train_duplicates_removed"] = dropped_train
        print(f"🗑️  Intra-train duplicates removed:  {dropped_train}")

        # --- Step 2: remove from val/test any hash that appears in train
        train_hashes  = set(df.loc[df["split"] == "train", "sha256"])
        val_test_mask = df["split"].isin(["val", "test"])
        leaked        = val_test_mask & df["sha256"].isin(train_hashes)
        dropped_leak  = int(leaked.sum())
        df = df[~leaked]
        report["train_val_test_leakage_removed"] = dropped_leak
        print(f"🗑️  Train→val/test leakage removed:  {dropped_leak}")

        # --- Step 3: remove duplicates between val and test (keep in val)
        val_hashes  = set(df.loc[df["split"] == "val", "sha256"])
        test_mask   = df["split"] == "test"
        val_test_dup = test_mask & df["sha256"].isin(val_hashes)
        dropped_vt   = int(val_test_dup.sum())
        df = df[~val_test_dup]
        report["val_test_duplicates_removed"] = dropped_vt
        print(f"🗑️  Val↔test duplicates removed:     {dropped_vt}")

        # --- Assertion: zero overlap
        train_h = set(df.loc[df["split"] == "train", "sha256"])
        val_h   = set(df.loc[df["split"] == "val",   "sha256"])
        test_h  = set(df.loc[df["split"] == "test",  "sha256"])
        assert len(train_h & val_h)  == 0, "ASSERT FAILED: train∩val overlap!"
        assert len(train_h & test_h) == 0, "ASSERT FAILED: train∩test overlap!"
        assert len(val_h   & test_h) == 0, "ASSERT FAILED: val∩test overlap!"
        print("✅ Zero hash overlap between all splits confirmed.")

        report["total_removed"] = initial - len(df)
        report["remaining"]     = len(df)
        return df, report

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Resolution / format report
    # ─────────────────────────────────────────────────────────────────────────

    def resolution_format_report(
        self, df: pd.DataFrame, dataset_name: str
    ) -> Dict:
        """Per-class resolution and format statistics; flags potential shortcuts."""
        cfg = self.cfg
        report: Dict = {"dataset": dataset_name, "per_class": {}}

        valid = df[(df["orig_width"] > 0) & (df["orig_height"] > 0)].copy()

        for label_val, label_name in [(0, "real"), (1, "fake")]:
            subset = valid[valid["label"] == label_val]
            if subset.empty:
                report["per_class"][label_name] = {}
                continue
            stats = {
                "count": int(len(subset)),
                "width":  {
                    "min":    int(subset["orig_width"].min()),
                    "median": int(subset["orig_width"].median()),
                    "max":    int(subset["orig_width"].max()),
                },
                "height": {
                    "min":    int(subset["orig_height"].min()),
                    "median": int(subset["orig_height"].median()),
                    "max":    int(subset["orig_height"].max()),
                },
                "format_share": subset["orig_format"].value_counts(normalize=True)
                                       .mul(100).round(1).to_dict(),
            }
            report["per_class"][label_name] = stats

        # --- Check for suspicious real-vs-fake mismatches
        warnings_list: List[str] = []
        real_s = report["per_class"].get("real", {})
        fake_s = report["per_class"].get("fake", {})
        if real_s and fake_s:
            real_med_w = real_s.get("width",  {}).get("median", 0)
            fake_med_w = fake_s.get("width",  {}).get("median", 0)
            real_med_h = real_s.get("height", {}).get("median", 0)
            fake_med_h = fake_s.get("height", {}).get("median", 0)

            if abs(real_med_w - fake_med_w) > 32:
                msg = (f"⚠️  Median width mismatch: real={real_med_w}px "
                       f"vs fake={fake_med_w}px — potential resolution shortcut!")
                warnings_list.append(msg)
                print(msg)
            if abs(real_med_h - fake_med_h) > 32:
                msg = (f"⚠️  Median height mismatch: real={real_med_h}px "
                       f"vs fake={fake_med_h}px — potential resolution shortcut!")
                warnings_list.append(msg)
                print(msg)

            # Format mismatch check
            real_top_fmt = max(real_s.get("format_share", {"?": 0}),
                               key=real_s.get("format_share", {"?": 0}).get, default="?")
            fake_top_fmt = max(fake_s.get("format_share", {"?": 0}),
                               key=fake_s.get("format_share", {"?": 0}).get, default="?")
            if real_top_fmt != fake_top_fmt:
                msg = (f"⚠️  Dominant format mismatch: real={real_top_fmt} "
                       f"vs fake={fake_top_fmt} — potential format shortcut!")
                warnings_list.append(msg)
                print(msg)

        report["warnings"] = warnings_list
        if not warnings_list:
            print(f"✅ No suspicious resolution/format mismatches in {dataset_name}.")

        # Save JSON
        out_dir = Path(cfg.ROOT) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_resolution_report.json"
        with open(str(out_path), "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"💾 Resolution report saved to {out_path}")
        return report

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Subsample
    # ─────────────────────────────────────────────────────────────────────────

    def subsample(self, df: pd.DataFrame, cfg: "CFG") -> pd.DataFrame:
        """Class-balanced subsample up to max_samples_per_class per class per split."""
        seed = cfg.seeds[0]
        budget = cfg.quick_test_samples if cfg.QUICK_TEST else cfg.max_samples_per_class

        frames: List[pd.DataFrame] = []
        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]
            if split_df.empty:
                continue

            min_req = (cfg.val_min_samples  if split_name == "val"  else
                       cfg.test_min_samples if split_name == "test" else 0)

            sampled_parts: List[pd.DataFrame] = []
            for lbl in [0, 1]:
                cls_df = split_df[split_df["label"] == lbl]
                n_take = min(budget, len(cls_df))
                n_take = max(n_take, min(min_req, len(cls_df)))
                if len(cls_df) > n_take:
                    cls_sample = cls_df.sample(n=n_take, random_state=seed)
                else:
                    cls_sample = cls_df
                sampled_parts.append(cls_sample)

            split_sampled = pd.concat(sampled_parts, ignore_index=True)
            frames.append(split_sampled)
            print(f"   {split_name:5s}: {len(split_sampled):>6d} files after subsample "
                  f"({len(split_df):>6d} before)")

        result = pd.concat(frames, ignore_index=True)
        print(f"✅ Total after subsample: {len(result)} files")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Center-crop and save
    # ─────────────────────────────────────────────────────────────────────────

    def center_crop_and_save(
        self, df: pd.DataFrame, dataset_name: str, cfg: "CFG"
    ) -> pd.DataFrame:
        """Center-crop (PIL crop only, no resize) and save as lossless PNG."""
        crop_size = cfg.dataset_crop_sizes.get(dataset_name, cfg.common_crop)
        out_root  = Path(cfg.SUBSET_ROOT) / dataset_name
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"\n✂️  Center-cropping {dataset_name} to {crop_size}×{crop_size} px …")
        print(f"   Output directory: {out_root}")

        rel_paths: List[Optional[str]] = []
        crop_sizes: List[int] = []
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc="Cropping", unit="img"):
            src_path = row["filepath"]
            split    = row.get("split", "train")
            label    = row["label"]

            split_dir = out_root / str(split) / str(label)
            split_dir.mkdir(parents=True, exist_ok=True)

            # Destination filename (lossless PNG)
            stem     = Path(src_path).stem
            dst_name = f"{stem}.png"
            dst_path = split_dir / dst_name

            # Relative path stored in manifest
            rel = str(dst_path.relative_to(Path(cfg.SUBSET_ROOT)))

            if dst_path.exists():
                rel_paths.append(rel)
                crop_sizes.append(crop_size)
                continue

            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    if w < crop_size or h < crop_size:
                        warnings.warn(
                            f"Image too small for crop ({w}×{h} < {crop_size}): "
                            f"{src_path}",
                            UserWarning, stacklevel=2,
                        )
                        skipped += 1
                        rel_paths.append(None)
                        crop_sizes.append(crop_size)
                        continue

                    # Centre crop (no interpolation)
                    left   = (w - crop_size) // 2
                    top    = (h - crop_size) // 2
                    right  = left + crop_size
                    bottom = top  + crop_size
                    cropped = img.crop((left, top, right, bottom))
                    cropped.save(str(dst_path), format="PNG", compress_level=1)

                rel_paths.append(rel)
                crop_sizes.append(crop_size)

            except Exception as exc:
                warnings.warn(f"Failed to crop {src_path}: {exc}", UserWarning, stacklevel=2)
                skipped += 1
                rel_paths.append(None)
                crop_sizes.append(crop_size)

        df = df.copy()
        df["rel_path"]  = rel_paths
        df["crop_size"] = crop_sizes

        # Drop rows that could not be cropped
        before = len(df)
        df = df[df["rel_path"].notna()].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"⚠️  Dropped {dropped} images that could not be cropped.")
        print(f"✅ Cropped {len(df)} images.  Skipped: {skipped}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Save manifest
    # ─────────────────────────────────────────────────────────────────────────

    def save_manifest(
        self, df: pd.DataFrame, dataset_name: str, cfg: "CFG"
    ) -> str:
        """Save CSV manifest to ROOT/manifests/<dataset_name>_manifest.csv."""
        manifest_dir = Path(cfg.ROOT) / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        out_path = manifest_dir / f"{dataset_name}_manifest.csv"

        cols = [
            "rel_path", "label", "split",
            "sha256", "orig_width", "orig_height", "orig_format", "crop_size",
        ]
        # Keep only columns that exist
        cols_present = [c for c in cols if c in df.columns]
        manifest_df  = df[cols_present].rename(columns={"rel_path": "path"})

        manifest_df.to_csv(str(out_path), index=False)
        print(f"💾 Manifest saved: {out_path}  ({len(manifest_df)} rows)")
        return str(out_path)

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Cleanup raw data
    # ─────────────────────────────────────────────────────────────────────────

    def cleanup_raw(self, dataset_name: str, raw_dir: str) -> None:
        """Delete raw zip and extracted raw folders after pipeline is done."""
        raw_path = Path(raw_dir)
        if raw_path.exists():
            shutil.rmtree(str(raw_path), ignore_errors=True)
            print(f"🗑️  Deleted raw directory: {raw_path}")

        # Also remove any zip files that might linger in /content/
        for zip_file in Path("/content").glob(f"*.zip"):
            zip_file.unlink(missing_ok=True)
            print(f"🗑️  Deleted zip: {zip_file}")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. Full pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def process_dataset(self, dataset_name: str) -> str:
        """Run the full pipeline for one dataset.  Returns the manifest path."""
        cfg = self.cfg
        print(f"\n{'#'*60}")
        print(f"# Processing dataset: {dataset_name}")
        print(f"{'#'*60}")

        # Drive space check before each dataset
        check_drive_space(cfg.min_drive_free_gb, cfg)

        # 1. Download
        raw_dir = self.download_dataset(dataset_name)

        # 2. Inspect + label
        file_list, label_map = self.build_label_map(dataset_name, raw_dir)

        # 3. Visual confirmation grid (4x4 real vs. AI)
        try:
            inv_map = {0: "Real", 1: "AI-Generated"}
            show_sample_grid(
                paths=[f[0] for f in file_list],
                labels=[f[1] for f in file_list],
                label_map_inv=inv_map,
                title=f"{dataset_name} — Real vs AI Sample Grid",
                n=4,
                cfg=cfg,
            )
        except Exception as e:
            print(f"⚠️  Could not display sample grid: {e}")

        # 4. Hash + metadata
        df = self.compute_hashes_and_info(file_list)

        # Drop hashing errors
        error_mask = df["sha256"] == "ERROR"
        if error_mask.any():
            print(f"⚠️  Dropping {error_mask.sum()} files that could not be hashed.")
            df = df[~error_mask].reset_index(drop=True)

        # Drop unreadable images
        invalid_img = df["orig_width"] == -1
        if invalid_img.any():
            print(f"⚠️  Dropping {invalid_img.sum()} unreadable image files.")
            df = df[~invalid_img].reset_index(drop=True)

        # 4. Assign splits
        df = self.assign_splits(df)

        # 5. Dedup
        df, dedup_report = self.deduplicate(df)
        print(f"   Dedup report: {dedup_report}")
        dedup_dir = Path(cfg.ROOT) / "results"
        dedup_dir.mkdir(parents=True, exist_ok=True)
        with open(str(dedup_dir / f"{dataset_name}_dedup_report.json"), "w") as fh:
            json.dump(dedup_report, fh, indent=2)

        # 6. Resolution / format report
        self.resolution_format_report(df, dataset_name)

        # 7. Subsample
        print(f"\n📊 Subsampling {dataset_name} …")
        df = self.subsample(df, cfg)

        # Validate minimum counts
        for split_name, min_req in [("val", cfg.val_min_samples),
                                    ("test", cfg.test_min_samples)]:
            cnt = (df["split"] == split_name).sum()
            if cnt < min_req:
                raise RuntimeError(
                    f"{dataset_name} {split_name} split has only {cnt} samples "
                    f"(minimum required: {min_req}).  Consider lowering "
                    f"cfg.{split_name}_min_samples or increasing the dataset budget."
                )

        # 8. Center-crop and save
        df = self.center_crop_and_save(df, dataset_name, cfg)

        # 9. Save manifest
        manifest_path = self.save_manifest(df, dataset_name, cfg)
        cfg.manifests[dataset_name] = manifest_path
        cfg.subset_roots[dataset_name] = cfg.SUBSET_ROOT

        # 10. Save label_map to configs/
        cfg_dir = Path(cfg.ROOT) / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        lmap_path = cfg_dir / f"{dataset_name}_label_map.json"
        with open(str(lmap_path), "w") as fh:
            json.dump(label_map, fh, indent=2)
        print(f"💾 Label map saved to {lmap_path}")

        # 11. Cleanup raw
        if cfg.delete_after_training or cfg.process_per_dataset:
            self.cleanup_raw(dataset_name, raw_dir)

        print(f"\n✅ {dataset_name} pipeline complete.  Manifest: {manifest_path}")
        return manifest_path


print("✅ DownloadManager class defined.")

# ===CELL_BREAK===

# [MARKDOWN]
# ## Download & Process Datasets (DownloadManager Pipeline)
# Runs the DownloadManager pipeline for each dataset in `cfg.datasets` in strict sequential order:
# 1. Downloads & unzips via Kaggle CLI
# 2. Inspects directory tree & derives explicit `label_map` (raises `ValueError` on ambiguity)
# 3. Displays 4x4 visual confirmation grid (real vs. AI)
# 4. Computes chunked SHA-256 and records original image dimensions/format
# 5. Preserves existing splits or creates stratified 70/15/15 splits
# 6. Performs intra-split & cross-split deduplication (zero hash overlap asserted)
# 7. Generates resolution & format shortcut reports (flags mismatches)
# 8. Subsamples class-balanced splits up to `cfg.max_samples_per_class`
# 9. Center-crops with NO interpolation to lossless PNGs in `/content/subsets/<dataset>/`
# 10. Exports CSV manifest with relative paths to Drive `ROOT/manifests/`
# 11. Cleans up raw zip and unzipped folders before moving to the next dataset

# ===CELL_BREAK===

# [CODE]
# ── Execute Download & Preprocessing Pipeline for All Configured Datasets ───
dm = DownloadManager(cfg)

for ds in cfg.datasets:
    manifest_file = Path(cfg.ROOT) / "manifests" / f"{ds}_manifest.csv"
    subset_dir = Path(cfg.SUBSET_ROOT) / ds

    # If dataset has already been processed on local disk + manifest on Drive, reuse it
    if manifest_file.exists() and subset_dir.exists():
        print(f"\nℹ️  {ds} already downloaded and processed. Using existing manifest:")
        print(f"   {manifest_file}")
        cfg.manifests[ds] = str(manifest_file)
        cfg.subset_roots[ds] = cfg.SUBSET_ROOT
    else:
        print(f"\n{'='*60}")
        print(f"🚀 Processing dataset: {ds}")
        print(f"{'='*60}")
        mpath = dm.process_dataset(ds)
        cfg.manifests[ds] = mpath
        cfg.subset_roots[ds] = cfg.SUBSET_ROOT

print("\n" + "=" * 60)
print("All datasets processed. Registered manifests:")
for k, v in cfg.manifests.items():
    print(f"  {k:12s} -> {v}")
print("=" * 60)
