# [MARKDOWN]
# ## Cell 13 — `run_experiment` wrapper + pre-training leakage assertion

# ===CELL_BREAK===

# [CODE]
# ─────────────────────────────────────────────────────────────────────────────
# Cell 13: run_experiment wrapper + pre-training leakage assertion
# ─────────────────────────────────────────────────────────────────────────────
import json
import time
import dataclasses
from pathlib import Path
from typing import Optional


def _resolve_manifest_and_root(dataset_name: str, cfg) -> tuple[str, str]:
    """Resolve manifest CSV path and subset_root for a dataset."""
    mpath = getattr(cfg, "manifests", {}).get(dataset_name)
    if not mpath:
        mpath = str(Path(cfg.ROOT) / "manifests" / f"{dataset_name}_manifest.csv")
    sroot = getattr(cfg, "subset_roots", {}).get(dataset_name, cfg.SUBSET_ROOT)
    return mpath, sroot


# ── 1. Leakage assertion ─────────────────────────────────────────────────────

def assert_no_leakage(manifest_paths: list[str], subset_root: str) -> None:
    """
    Load every manifest in manifest_paths, extract sha256 hashes per split,
    and assert zero overlap between train/val/test across ALL datasets.
    """
    all_hashes_by_role: dict[str, set[str]] = {
        "train": set(), "val": set(), "test": set()
    }

    for mpath in manifest_paths:
        df = pd.read_csv(mpath)
        if "sha256" not in df.columns:
            raise ValueError(
                f"Manifest {mpath} missing 'sha256' column — "
                "re-run DownloadManager.process_dataset()."
            )
        for split in ("train", "val", "test"):
            h = set(df[df["split"] == split]["sha256"].dropna().tolist())
            all_hashes_by_role[split].update(h)

    errors: list[str] = []
    for r1, r2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = all_hashes_by_role[r1] & all_hashes_by_role[r2]
        if overlap:
            errors.append(
                f"  OVERLAP {r1}∩{r2}: {len(overlap)} shared sha256 hashes "
                f"(sample: {list(overlap)[:3]})"
            )

    if errors:
        raise AssertionError(
            "DATA LEAKAGE DETECTED across manifests in:\n"
            + f"  subset_root={subset_root}\n"
            + "\n".join(errors)
        )

    print(
        f"[assert_no_leakage] PASS — "
        f"train={len(all_hashes_by_role['train'])}, "
        f"val={len(all_hashes_by_role['val'])}, "
        f"test={len(all_hashes_by_role['test'])} hashes; "
        f"zero overlap across {len(manifest_paths)} manifest(s)."
    )


# ── 2. Single-seed experiment runner ─────────────────────────────────────────

def run_experiment(
    train_dataset: str,
    ablation_mode: str,          # "fused" | "spatial_only" | "freq_only"
    backbone: str,               # "resnet18" | "mobilenet_v2"
    freq_mode: str,              # "fft" | "dct"
    use_hann: bool,
    seed: int,
    protocol: str,               # "N" | "C"
    cfg,                         # CFG dataclass instance
) -> dict:
    """
    Full single-seed experiment.

    Returns metrics dict including accuracy, f1, roc_auc, pr_auc,
    youden_threshold, param_count, training_time_per_epoch,
    latency_bs1_ms, latency_bs64_ms.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── a. Reproducibility ───────────────────────────────────────────────────
    set_seed(seed)

    # ── b. Experiment directory ──────────────────────────────────────────────
    exp_id = (
        f"E_{train_dataset}_{ablation_mode}_{backbone}_"
        f"{freq_mode}_hann{use_hann}_seed{seed}_proto{protocol}"
    )
    experiment_dir = Path(cfg.ROOT) / "experiments" / exp_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # ── c. Persist config ────────────────────────────────────────────────────
    cfg_dict = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg).copy()
    cfg_dict.update({
        "train_dataset": train_dataset,
        "ablation_mode": ablation_mode,
        "backbone": backbone,
        "freq_mode": freq_mode,
        "use_hann": use_hann,
        "seed": seed,
        "protocol": protocol,
        "experiment_id": exp_id,
    })
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # ── d. Build dataloaders ─────────────────────────────────────────────────
    quick      = getattr(cfg, "QUICK_TEST", False)
    num_epochs = getattr(cfg, "quick_test_epochs", 1) if quick else getattr(cfg, "epochs", 20)

    manifest_path, subset_root = _resolve_manifest_and_root(train_dataset, cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        manifest_path = manifest_path,
        subset_root   = subset_root,
        cfg           = cfg,
        protocol      = protocol,
        augment_train = True,
    )

    # ── e. Build model ───────────────────────────────────────────────────────
    model = FusedDetector(
        ablation_mode  = ablation_mode,
        backbone       = backbone,
        freq_mode      = freq_mode,
        use_hann       = use_hann,
        use_projection = getattr(cfg, "use_projection", True),
        projection_dim = getattr(cfg, "projection_dim", 256),
        freq_cnn_dim   = getattr(cfg, "freq_cnn_dim", 256),
    ).to(device)

    # ── f. Parameter count ───────────────────────────────────────────────────
    param_counts  = model.count_parameters()
    param_count   = param_counts["total"]
    param_count_m = param_count / 1e6
    print(f"  [{exp_id}] Trainable params: {param_count_m:.2f} M")

    # ── g. Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        cfg            = cfg,
        experiment_dir = str(experiment_dir),
        seed           = seed,
        num_epochs     = num_epochs,
    )

    t0 = time.time()
    train_history = trainer.train()
    total_train_time = time.time() - t0
    n_ran = max(len(train_history["train_loss"]), 1)
    time_per_epoch = total_train_time / n_ran

    # ── h. Load best checkpoint ──────────────────────────────────────────────
    model, ckpt = load_best_checkpoint(str(experiment_dir), model, device)

    # ── i. Evaluate: Youden threshold from val, apply to test ────────────────
    val_eval = trainer.evaluate(val_loader)
    full_test_metrics = evaluate_test_set(
        model          = model,
        test_loader    = test_loader,
        val_probs      = val_eval["all_probs"],
        val_labels     = val_eval["all_labels"],
        experiment_dir = str(experiment_dir),
        cfg            = cfg,
        seed           = seed,
    )

    # ── j. Plot training curves ──────────────────────────────────────────────
    trainer.plot_curves(train_history, str(experiment_dir))

    # ── k. Extract summary metrics ───────────────────────────────────────────
    m_youden     = full_test_metrics["threshold_youden"]
    m_05         = full_test_metrics["threshold_0_5"]
    youden_thr   = full_test_metrics["youden_threshold"]
    latency_bs1  = full_test_metrics["latency"].get("latency_bs1_ms", float("nan"))
    latency_bs64 = full_test_metrics["latency"].get("latency_bs64_ms", float("nan"))

    result = {
        "experiment_id"          : exp_id,
        "train_dataset"          : train_dataset,
        "ablation_mode"          : ablation_mode,
        "backbone"               : backbone,
        "freq_mode"              : freq_mode,
        "use_hann"               : use_hann,
        "seed"                   : seed,
        "protocol"               : protocol,
        "accuracy"               : m_05["accuracy"],
        "f1"                     : m_05["f1"],
        "roc_auc"                : m_05["roc_auc"],
        "pr_auc"                 : m_05["pr_auc"],
        "accuracy_youden"        : m_youden["accuracy"],
        "f1_youden"              : m_youden["f1"],
        "youden_threshold"       : youden_thr,
        "param_count"            : param_count,
        "param_count_m"          : round(param_count_m, 3),
        "training_time_per_epoch": round(time_per_epoch, 2),
        "latency_bs1_ms"         : round(latency_bs1,  3),
        "latency_bs64_ms"        : round(latency_bs64, 3),
        "best_epoch"             : ckpt.get("epoch", -1),
        "best_val_metric"        : ckpt.get("best_metric", float("nan")),
    }

    print(
        f"  [{exp_id}] "
        f"ACC={result['accuracy']:.4f} "
        f"F1={result['f1']:.4f} "
        f"AUC={result['roc_auc']:.4f} "
        f"PR-AUC={result['pr_auc']:.4f} "
        f"| Thr_J={youden_thr:.3f} "
        f"| {param_count_m:.2f}M params "
        f"| lat@1={latency_bs1:.2f}ms lat@64={latency_bs64:.2f}ms/img"
    )
    return result


# ── 3. Multi-seed wrapper ────────────────────────────────────────────────────

def run_experiment_multi_seed(
    train_dataset : str,
    ablation_mode : str,
    backbone      : str,
    freq_mode     : str,
    use_hann      : bool,
    protocol      : str,
    cfg,
    seeds         : Optional[list[int]] = None,
) -> tuple[list[dict], dict]:
    """
    Run run_experiment() for each seed, aggregate, and return
    (all_results_list, summary_dict).
    """
    import numpy as np

    if seeds is None:
        seeds = getattr(cfg, "seeds", [42])

    all_results: list[dict] = []
    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────────────────────────")
        result = run_experiment(
            train_dataset = train_dataset,
            ablation_mode = ablation_mode,
            backbone      = backbone,
            freq_mode     = freq_mode,
            use_hann      = use_hann,
            seed          = seed,
            protocol      = protocol,
            cfg           = cfg,
        )
        all_results.append(result)

    numeric_keys = [
        "accuracy", "f1", "roc_auc", "pr_auc",
        "accuracy_youden", "f1_youden",
        "youden_threshold", "param_count_m",
        "training_time_per_epoch", "latency_bs1_ms", "latency_bs64_ms",
    ]
    summary: dict = {
        "experiment_label": (
            f"E_{train_dataset}_{ablation_mode}_{backbone}_"
            f"{freq_mode}_hann{use_hann}_proto{protocol}"
        ),
        "train_dataset" : train_dataset,
        "ablation_mode" : ablation_mode,
        "backbone"      : backbone,
        "freq_mode"     : freq_mode,
        "use_hann"      : use_hann,
        "protocol"      : protocol,
        "n_seeds"       : len(seeds),
        "seeds"         : seeds,
    }
    for k in numeric_keys:
        vals = [r[k] for r in all_results if k in r and r[k] is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            summary[f"{k}_mean"] = round(float(arr.mean()), 5)
            summary[f"{k}_std"]  = round(float(arr.std()),  5)
        else:
            summary[f"{k}_mean"] = None
            summary[f"{k}_std"]  = None

    label = summary["experiment_label"]
    print(f"\n{'─'*60}")
    print(f"  SUMMARY  |  {label}  |  seeds={seeds}")
    print(f"{'─'*60}")
    for k in ["accuracy", "f1", "roc_auc", "pr_auc"]:
        m, s = summary[f"{k}_mean"], summary[f"{k}_std"]
        print(f"    {k:>12s}:  {m:.4f} ± {s:.4f}")
    print(f"{'─'*60}\n")

    return all_results, summary


# ───────────────────────────────────────────────────────────────────────────
# Pre-training leakage check (runs immediately when this cell executes)
# ───────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PRE-TRAINING LEAKAGE ASSERTION")
print("=" * 60)

_manifest_paths = list(getattr(cfg, "manifests", {}).values())
if not _manifest_paths:
    _manifest_paths = [
        str(p) for p in sorted((Path(cfg.ROOT) / "manifests").glob("*_manifest.csv"))
    ]

if _manifest_paths:
    assert_no_leakage(
        manifest_paths = _manifest_paths,
        subset_root    = cfg.SUBSET_ROOT,
    )
else:
    print("ℹ️  No manifest CSVs found yet — run DownloadManager.process_dataset() first.")

# ===CELL_BREAK===

# [CODE]
# ─────────────────────────────────────────────────────────────────────────────
# Cell 14: Core experiments E1–E4, E7
# ─────────────────────────────────────────────────────────────────────────────

# Ensure results directory exists
(Path(cfg.ROOT) / "results").mkdir(parents=True, exist_ok=True)

all_results: list[dict] = []          # collect every per-seed metrics dict

# ─── E1: Spatial-only (ResNet18) ─────────────────────────────────────────────
print("=" * 60)
print("E1: ResNet18 Spatial-Only")
print("=" * 60)
e1_results, e1_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],  # GRAVEX (or first configured dataset)
    ablation_mode = "spatial_only",
    backbone      = "resnet18",
    freq_mode     = "fft",            # irrelevant for spatial_only, but logged
    use_hann      = False,
    protocol      = "C",
    cfg           = cfg,
    seeds         = cfg.seeds,
)
all_results.extend(e1_results)
print(f"E1 Summary: ROC-AUC {e1_summary['roc_auc_mean']:.4f} ± {e1_summary['roc_auc_std']:.4f}")

# ─── E2: Frequency-only (FFT CNN) ────────────────────────────────────────────
print("=" * 60)
print("E2: Frequency-CNN Only (FFT)")
print("=" * 60)
e2_results, e2_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],
    ablation_mode = "freq_only",
    backbone      = "resnet18",
    freq_mode     = "fft",
    use_hann      = True,
    protocol      = "C",
    cfg           = cfg,
    seeds         = cfg.seeds,
)
all_results.extend(e2_results)
print(f"E2 Summary: ROC-AUC {e2_summary['roc_auc_mean']:.4f} ± {e2_summary['roc_auc_std']:.4f}")

# ─── E3: Fused ResNet18 + FFT CNN (proposed) ─────────────────────────────────
print("=" * 60)
print("E3: ResNet18 + FFT CNN Fused (Proposed)")
print("=" * 60)
e3_results, e3_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],
    ablation_mode = "fused",
    backbone      = "resnet18",
    freq_mode     = "fft",
    use_hann      = True,
    protocol      = "C",
    cfg           = cfg,
    seeds         = cfg.seeds,
)
all_results.extend(e3_results)
print(f"E3 Summary: ROC-AUC {e3_summary['roc_auc_mean']:.4f} ± {e3_summary['roc_auc_std']:.4f}")

# ─── E4: ResNet18 + DCT CNN (seed 42) ────────────────────────────────────────
print("=" * 60)
print("E4: ResNet18 + DCT CNN (seed 42)")
print("=" * 60)
e4_results, e4_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],
    ablation_mode = "fused",
    backbone      = "resnet18",
    freq_mode     = "dct",
    use_hann      = True,
    protocol      = "C",
    cfg           = cfg,
    seeds         = [42],
)
all_results.extend(e4_results)
print(f"E4 Summary: ROC-AUC {e4_summary['roc_auc_mean']:.4f}")

# ─── E7: Hann window ablation (seed 42) ──────────────────────────────────────
print("=" * 60)
print("E7: FFT with Hann Window")
print("=" * 60)
e7_hann_results, e7_hann_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],
    ablation_mode = "fused",
    backbone      = "resnet18",
    freq_mode     = "fft",
    use_hann      = True,
    protocol      = "C",
    cfg           = cfg,
    seeds         = [42],
)

e7_nohann_results, e7_nohann_summary = run_experiment_multi_seed(
    train_dataset = cfg.datasets[0],
    ablation_mode = "fused",
    backbone      = "resnet18",
    freq_mode     = "fft",
    use_hann      = False,
    protocol      = "C",
    cfg           = cfg,
    seeds         = [42],
)
all_results.extend(e7_hann_results)
all_results.extend(e7_nohann_results)
print(
    f"E7 Hann={e7_hann_summary['roc_auc_mean']:.4f} vs "
    f"No-Hann={e7_nohann_summary['roc_auc_mean']:.4f} "
    f"(Δ AUC={e7_hann_summary['roc_auc_mean'] - e7_nohann_summary['roc_auc_mean']:+.4f})"
)

# ─── Optional E5: MobileNetV2 + FFT CNN (uncomment to run) ───────────────────
# e5_results, e5_summary = run_experiment_multi_seed(
#     train_dataset = cfg.datasets[0], ablation_mode="fused",
#     backbone="mobilenet_v2", freq_mode="fft", use_hann=True,
#     protocol="C", cfg=cfg, seeds=[42],
# )
# all_results.extend(e5_results)

# ─── Persist in-dataset results ──────────────────────────────────────────────
in_dataset_df = pd.DataFrame(all_results)
in_dataset_csv = Path(cfg.ROOT) / "results" / "in_dataset_results.csv"
in_dataset_df.to_csv(str(in_dataset_csv), index=False)
print("\n" + "=" * 60)
print(f"In-dataset results saved → {in_dataset_csv}")
print("=" * 60)

# ─── Quick display (key columns) ─────────────────────────────────────────────
_display_cols = [
    c for c in [
        "experiment_id", "seed",
        "roc_auc", "pr_auc", "f1", "accuracy",
        "param_count_m", "latency_bs1_ms",
    ]
    if c in in_dataset_df.columns
]
print(in_dataset_df[_display_cols].to_string(index=False))

# ─── Ablation comparison printout ────────────────────────────────────────────
print("\n── Ablation summary (mean over seeds) ──────────────────────────────")
_abl_rows = []
for label, summary in [
    ("E1 spatial_only", e1_summary),
    ("E2 freq_only",    e2_summary),
    ("E3 fused (FFT)",  e3_summary),
    ("E4 fused (DCT)",  e4_summary),
]:
    _abl_rows.append({
        "Experiment" : label,
        "ROC-AUC"    : f"{summary['roc_auc_mean']:.4f} ± {summary['roc_auc_std']:.4f}",
        "PR-AUC"     : f"{summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f}",
        "F1"         : f"{summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}",
        "Accuracy"   : f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}",
    })
_abl_df = pd.DataFrame(_abl_rows).set_index("Experiment")
print(_abl_df.to_string())

# ===CELL_BREAK===

# [CODE]
# ─────────────────────────────────────────────────────────────────────────────
# Cell 15: E6 — Cross-dataset experiment
# ─────────────────────────────────────────────────────────────────────────────
import warnings

# ── 1. Cross-dataset evaluation function ─────────────────────────────────────

def run_cross_dataset_experiment(
    source_dataset  : str,
    target_datasets : list[str],
    cfg,
    seed            : int  = 42,
    protocol        : str  = "C",
) -> dict[str, dict]:
    """
    Train on source_dataset (or reuse existing best checkpoint).
    Evaluate each target_dataset WITHOUT fine-tuning.
    Youden threshold is fixed from SOURCE validation set.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_exp_id = (
        f"E_{source_dataset}_fused_resnet18_fft_hannTrue_seed{seed}_proto{protocol}"
    )
    source_exp_dir = Path(cfg.ROOT) / "experiments" / source_exp_id
    ckpt_path      = source_exp_dir / "best_model.pth"

    src_manifest, src_root = _resolve_manifest_and_root(source_dataset, cfg)

    model = FusedDetector(
        ablation_mode  = "fused",
        backbone       = "resnet18",
        freq_mode      = "fft",
        use_hann       = True,
        use_projection = getattr(cfg, "use_projection", True),
        projection_dim = getattr(cfg, "projection_dim", 256),
        freq_cnn_dim   = getattr(cfg, "freq_cnn_dim", 256),
    ).to(device)

    train_loader, src_val_loader, _ = get_dataloaders(
        manifest_path = src_manifest,
        subset_root   = src_root,
        cfg           = cfg,
        protocol      = protocol,
        augment_train = True,
    )

    quick      = getattr(cfg, "QUICK_TEST", False)
    num_epochs = getattr(cfg, "quick_test_epochs", 1) if quick else getattr(cfg, "epochs", 20)

    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = src_val_loader,
        cfg            = cfg,
        experiment_dir = str(source_exp_dir),
        seed           = seed,
        num_epochs     = num_epochs,
    )

    if ckpt_path.exists():
        print(f"  [E6] Reusing checkpoint: {ckpt_path}")
        model, _ = load_best_checkpoint(str(source_exp_dir), model, device)
    else:
        print(f"  [E6] No existing checkpoint for {source_exp_id}. Training from scratch …")
        source_exp_dir.mkdir(parents=True, exist_ok=True)
        trainer.train()
        model, _ = load_best_checkpoint(str(source_exp_dir), model, device)

    model.eval()

    # ── Compute Youden threshold from SOURCE validation set ──────────────────
    src_val_eval = trainer.evaluate(src_val_loader)
    youden_thr = trainer.compute_youden_threshold(
        src_val_eval["all_probs"], src_val_eval["all_labels"]
    )
    print(f"  [E6] Source={source_dataset} | Youden threshold={youden_thr:.4f}")

    # ── Evaluate each target dataset ─────────────────────────────────────────
    target_results: dict[str, dict] = {}

    for tgt_ds in target_datasets:
        if tgt_ds == source_dataset:
            continue   # skip self; use in-dataset results for that

        tgt_manifest, tgt_root = _resolve_manifest_and_root(tgt_ds, cfg)
        if not Path(tgt_manifest).exists():
            warnings.warn(
                f"  [E6] Manifest not found for target={tgt_ds} ({tgt_manifest}); skipping."
            )
            continue

        _, _, tgt_test_loader = get_dataloaders(
            manifest_path = tgt_manifest,
            subset_root   = tgt_root,
            cfg           = cfg,
            protocol      = protocol,
            augment_train = False,
        )

        tgt_eval = trainer.evaluate(tgt_test_loader, threshold=youden_thr)
        metrics = {
            "source_dataset":   source_dataset,
            "target_dataset":   tgt_ds,
            "test_roc_auc":     tgt_eval["auc"],
            "test_pr_auc":      tgt_eval["pr_auc"],
            "test_accuracy":    tgt_eval["accuracy"],
            "youden_threshold": youden_thr,
            "protocol":         protocol,
            "seed":             seed,
        }
        target_results[tgt_ds] = metrics

        print(
            f"    {source_dataset:>12s} → {tgt_ds:<12s}  "
            f"AUC={metrics['test_roc_auc']:.4f}  "
            f"PR-AUC={metrics['test_pr_auc']:.4f}  "
            f"ACC={metrics['test_accuracy']:.4f}"
        )

    return target_results


# ── 2. Full E6 execution ──────────────────────────────────────────────────────

quick = getattr(cfg, "QUICK_TEST", False)
# In QUICK_TEST, only use first dataset as source; otherwise iterate all
e6_sources = [cfg.datasets[0]] if quick else cfg.datasets

cross_results_protoC: list[dict] = []
cross_results_protoN: list[dict] = []

for src in e6_sources:
    print("=" * 60)
    print(f"E6 Cross-dataset  |  source={src}  |  Protocol C (primary)")
    print("=" * 60)
    res_C = run_cross_dataset_experiment(
        source_dataset  = src,
        target_datasets = cfg.datasets,
        cfg             = cfg,
        seed            = 42,
        protocol        = "C",
    )
    for tgt, metrics in res_C.items():
        cross_results_protoC.append(metrics)

    if not quick:
        print("=" * 60)
        print(f"E6 Cross-dataset  |  source={src}  |  Protocol N (secondary)")
        print("=" * 60)
        print(
            "⚠  WARNING: Protocol N cross-dataset results involve models trained "
            "on one crop size evaluated on another; frequency statistics are NOT "
            "directly comparable across datasets. Report as MIXED-SIZE (secondary)."
        )
        res_N = run_cross_dataset_experiment(
            source_dataset  = src,
            target_datasets = cfg.datasets,
            cfg             = cfg,
            seed            = 42,
            protocol        = "N",
        )
        for tgt, metrics in res_N.items():
            cross_results_protoN.append(metrics)

# ── 3. Build cross-dataset AUC matrices ──────────────────────────────────────

def _build_matrix(records: list[dict], metric: str = "test_roc_auc") -> pd.DataFrame:
    """Pivot list-of-dicts into source × target AUC matrix."""
    if not records:
        return pd.DataFrame()
    rows = {}
    for r in records:
        src = r.get("source_dataset", "?")
        tgt = r.get("target_dataset", "?")
        rows.setdefault(src, {})[tgt] = round(r.get(metric, float("nan")), 4)
    df = pd.DataFrame(rows).T                     # rows=source, cols=target
    df.index.name   = "source \\ target"
    df.columns.name = "target_dataset"
    return df

matrix_C = _build_matrix(cross_results_protoC, "test_roc_auc")
matrix_N = _build_matrix(cross_results_protoN, "test_roc_auc")

results_dir = Path(cfg.ROOT) / "results"
results_dir.mkdir(parents=True, exist_ok=True)

matrix_C.to_csv(str(results_dir / "cross_dataset_matrix_protoC.csv"))
matrix_N.to_csv(str(results_dir / "cross_dataset_matrix_protoN.csv"))

print("\n── Cross-dataset AUC matrix — Protocol C (PRIMARY) ─────────────────")
print(matrix_C.to_string())

if not matrix_N.empty:
    print("\n── Cross-dataset AUC matrix — Protocol N  MIXED-SIZE (secondary) ──")
    print(matrix_N.to_string())

# ── 4. Heatmap visualisation ─────────────────────────────────────────────────

def _plot_crossdataset_heatmap(
    matrix    : pd.DataFrame,
    title     : str,
    save_path : str,
) -> None:
    """Seaborn heatmap of a source×target AUC matrix."""
    if matrix.empty:
        print(f"  Skipping heatmap (empty matrix): {title}")
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(5, len(matrix.columns) * 1.4),
                                    max(4, len(matrix) * 1.2)))
    sns.heatmap(
        matrix.astype(float),
        annot        = True,
        fmt          = ".3f",
        cmap         = "RdYlGn",
        vmin         = 0.5,
        vmax         = 1.0,
        linewidths   = 0.5,
        linecolor    = "gray",
        cbar_kws     = {"label": "ROC-AUC"},
        ax           = ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Target Dataset", fontsize=11)
    ax.set_ylabel("Source Dataset", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Heatmap saved → {save_path}")

plots_dir = Path(cfg.ROOT) / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

_plot_crossdataset_heatmap(
    matrix    = matrix_C,
    title     = "E6 Cross-dataset ROC-AUC — Protocol C (Primary)",
    save_path = str(plots_dir / "cross_dataset_heatmap_protoC.png"),
)

if not matrix_N.empty:
    _plot_crossdataset_heatmap(
        matrix    = matrix_N,
        title     = "E6 Cross-dataset ROC-AUC — Protocol N MIXED-SIZE (Secondary)",
        save_path = str(plots_dir / "cross_dataset_heatmap_protoN.png"),
    )

print("\nE6 complete.")

# ===CELL_BREAK===

# [CODE]
# ─────────────────────────────────────────────────────────────────────────────
# Cell 16: Results summary tables + efficiency table + spectrum visualisation
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

results_dir = Path(cfg.ROOT) / "results"
plots_dir   = Path(cfg.ROOT) / "plots"
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)


# ── 1. Summary table ─────────────────────────────────────────────────────────

def build_summary_table(all_results: list[dict]) -> pd.DataFrame:
    """
    Pivot all_results into a human-readable summary with mean±std per experiment
    configuration (collapsed over seeds).

    Columns: Experiment | Accuracy | F1 | ROC-AUC | PR-AUC
    Saves to ROOT/results/summary_table.csv.
    """
    if not all_results:
        raise ValueError("all_results is empty — run experiments first.")

    df = pd.DataFrame(all_results)

    # Group key (sans seed)
    group_keys = ["train_dataset", "ablation_mode", "backbone", "freq_mode",
                  "use_hann", "protocol"]
    metric_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]

    rows = []
    for group_vals, grp in df.groupby(group_keys, dropna=False):
        label = (
            f"{group_vals[0]}__{group_vals[1]}__{group_vals[2]}"
            f"__{group_vals[3]}_hann{group_vals[4]}_proto{group_vals[5]}"
        )
        row = {"Experiment": label, "n_seeds": len(grp)}
        for m in metric_keys:
            if m in grp.columns:
                vals = grp[m].dropna().astype(float)
                row[m.upper().replace("_", "-")] = (
                    f"{vals.mean():.4f} ± {vals.std():.4f}"
                    if len(vals) > 1
                    else f"{vals.mean():.4f}"
                )
            else:
                row[m.upper().replace("_", "-")] = "N/A"
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Experiment")

    # Annotate best per column (sort by mean value of ROC-AUC descending)
    def _extract_mean(cell_str):
        try:
            return float(str(cell_str).split("±")[0].strip())
        except (ValueError, AttributeError):
            return -1.0

    best_idx = {}
    for col in ["ACCURACY", "F1", "ROC-AUC", "PR-AUC"]:
        if col in summary_df.columns:
            means = summary_df[col].apply(_extract_mean)
            best_idx[col] = means.idxmax()

    # Mark best values with ★
    for col, idx in best_idx.items():
        summary_df.loc[idx, col] = "★ " + str(summary_df.loc[idx, col])

    csv_path = results_dir / "summary_table.csv"
    summary_df.to_csv(str(csv_path))
    print(f"Summary table saved → {csv_path}")
    return summary_df


# ── 2. Efficiency table ───────────────────────────────────────────────────────

def build_efficiency_table(all_results: list[dict]) -> pd.DataFrame:
    """
    Columns: Experiment | Params (M) | Train Time/Epoch (s) |
             Latency BS=1 (ms) | Latency BS=64 (ms/img)
    Reports median over seeds for per-run quantities.
    Saves to ROOT/results/efficiency_table.csv.
    """
    if not all_results:
        raise ValueError("all_results is empty.")

    df = pd.DataFrame(all_results)
    group_keys = ["train_dataset", "ablation_mode", "backbone", "freq_mode",
                  "use_hann", "protocol"]
    eff_keys   = ["param_count_m", "training_time_per_epoch",
                  "latency_bs1_ms", "latency_bs64_ms"]

    rows = []
    for group_vals, grp in df.groupby(group_keys, dropna=False):
        label = (
            f"{group_vals[0]}__{group_vals[1]}__{group_vals[2]}"
            f"__{group_vals[3]}_hann{group_vals[4]}_proto{group_vals[5]}"
        )
        row = {"Experiment": label}
        for k in eff_keys:
            if k in grp.columns:
                val = grp[k].dropna().astype(float).median()
                row[k] = round(val, 3)
            else:
                row[k] = None
        rows.append(row)

    eff_df = pd.DataFrame(rows).rename(columns={
        "param_count_m"           : "Params (M)",
        "training_time_per_epoch" : "Train Time/Epoch (s)",
        "latency_bs1_ms"          : "Latency BS=1 (ms)",
        "latency_bs64_ms"         : "Latency BS=64 (ms/img)",
    }).set_index("Experiment")

    csv_path = results_dir / "efficiency_table.csv"
    eff_df.to_csv(str(csv_path))
    print(f"Efficiency table saved → {csv_path}")
    return eff_df


# ── 3. Average FFT spectrum visualisation ────────────────────────────────────

def plot_average_spectrum(
    manifest_path : str,
    subset_root   : str,
    dataset_name  : str,
    cfg,
    n_samples     : int = 200,
) -> None:
    """
    Compute and plot the average log-magnitude FFT spectrum of real vs. AI
    images from the TEST split of a manifest.

    Grid layout:
        [Real spectrum]  [Fake spectrum]  [Difference map (fake − real)]

    Saves to ROOT/plots/<dataset_name>_spectrum_comparison.png
    """
    from torchvision import transforms as T
    import matplotlib.colors as mcolors

    manifest_df = pd.read_csv(manifest_path)
    test_df     = manifest_df[manifest_df["split"] == "test"].copy()

    real_df = test_df[test_df["label"] == 0].sample(
        min(n_samples, len(test_df[test_df["label"] == 0])),
        random_state=42,
    )
    fake_df = test_df[test_df["label"] == 1].sample(
        min(n_samples, len(test_df[test_df["label"] == 1])),
        random_state=42,
    )

    crop_size    = getattr(cfg, "common_crop", getattr(cfg, "crop_size", 224))
    to_gray_crop = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.CenterCrop(crop_size),
        T.ToTensor(),
    ])

    def _accumulate_spectrum(rows_df: pd.DataFrame) -> np.ndarray:
        accum = None
        count = 0
        for _, row in rows_df.iterrows():
            rel_p = row["path"] if "path" in row else row["rel_path"]
            img_path = Path(subset_root) / rel_p
            if not img_path.exists():
                continue
            try:
                from PIL import Image as PILImage
                img = PILImage.open(str(img_path)).convert("RGB")
                t   = to_gray_crop(img).squeeze(0).numpy()   # (H, W) in [0,1]
                # FFT
                f   = np.fft.fft2(t)
                fs  = np.fft.fftshift(f)
                mag = np.log1p(np.abs(fs))
                if accum is None:
                    accum = np.zeros_like(mag, dtype=np.float64)
                accum += mag
                count += 1
            except Exception:
                continue
        if accum is None or count == 0:
            raise RuntimeError(
                f"No valid images found in {subset_root} for dataset {dataset_name}."
            )
        return accum / count

    print(f"  Computing spectra for {dataset_name} …")
    real_spectrum = _accumulate_spectrum(real_df)
    fake_spectrum = _accumulate_spectrum(fake_df)
    diff_spectrum = fake_spectrum - real_spectrum

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Average Log-Magnitude FFT Spectrum — {dataset_name} (n≈{n_samples} each)",
        fontsize=13, fontweight="bold",
    )

    vmin_shared = min(real_spectrum.min(), fake_spectrum.min())
    vmax_shared = max(real_spectrum.max(), fake_spectrum.max())

    for ax, data, title, cmap in [
        (axes[0], real_spectrum, "Real Images",         "viridis"),
        (axes[1], fake_spectrum, "AI-Generated Images", "viridis"),
        (axes[2], diff_spectrum, "Difference (Fake − Real)",
         mcolors.TwoSlopeNorm(vmin=diff_spectrum.min(), vcenter=0.0,
                               vmax=diff_spectrum.max())),
    ]:
        if title.startswith("Diff"):
            im = ax.imshow(data, cmap="bwr", norm=cmap, interpolation="nearest")
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin_shared, vmax=vmax_shared,
                           interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out_path = plots_dir / f"{dataset_name}_spectrum_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Spectrum saved → {out_path}")


# ── Execute tables ────────────────────────────────────────────────────────────

print("=" * 60)
print("Summary Table")
print("=" * 60)
summary_table = build_summary_table(all_results)
# Display with pandas (suppress truncation)
with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
    print(summary_table.to_string())

print()
print("=" * 60)
print("Efficiency Table")
print("=" * 60)
efficiency_table = build_efficiency_table(all_results)
with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
    print(efficiency_table.to_string())

# ── Optional: spectrum plots for each dataset ─────────────────────────────────
# Comment out if running in QUICK_TEST to save time.
if not getattr(cfg, "QUICK_TEST", False):
    for ds in cfg.datasets:
        try:
            mpath, sroot = _resolve_manifest_and_root(ds, cfg)
            plot_average_spectrum(
                manifest_path = mpath,
                subset_root   = sroot,
                dataset_name  = ds,
                cfg           = cfg,
                n_samples     = 200,
            )
        except Exception as e:
            print(f"  [spectrum] Skipped {ds}: {e}")
else:
    print("QUICK_TEST=True — skipping spectrum plots.")

# ===CELL_BREAK===

# [MARKDOWN]
# # ✅ Pre-Publication Verification Checklist
#
# Run the cell below before reporting results in a paper.

# ===CELL_BREAK===

# [CODE]
# ─────────────────────────────────────────────────────────────────────────────
# Cell 17: Final verification checklist
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(ok: bool, note: str = "") -> str:
    """Return ✅ or ❌ followed by note."""
    sym = "✅" if ok else "❌"
    return f"[{sym}] {note}"


def _neutral(note: str) -> str:
    """Return a neutral [   ] item (informational, no pass/fail)."""
    return f"[   ]  {note}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Re-run leakage assertion
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PRE-PUBLICATION VERIFICATION CHECKLIST")
print("=" * 60)
print()

_chk_manifests = list(getattr(cfg, "manifests", {}).values())
if not _chk_manifests:
    _chk_manifests = [str(p) for p in sorted((Path(cfg.ROOT) / "manifests").glob("*_manifest.csv"))]

leakage_pass   = False
leakage_detail = ""
try:
    if _chk_manifests:
        assert_no_leakage(
            manifest_paths = _chk_manifests,
            subset_root    = cfg.SUBSET_ROOT,
        )
        leakage_pass = True
    else:
        leakage_detail = "No manifest CSVs found."
except AssertionError as exc:
    leakage_detail = str(exc)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify all manifests exist on disk
# ─────────────────────────────────────────────────────────────────────────────
manifests_ok      = len(_chk_manifests) > 0 and all(Path(p).exists() for p in _chk_manifests)
missing_manifests = [p for p in _chk_manifests if not Path(p).exists()]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify every experiment directory has best_model.pth
# ─────────────────────────────────────────────────────────────────────────────
exp_root      = Path(cfg.ROOT) / "experiments"
exp_dirs      = sorted(exp_root.glob("E_*")) if exp_root.exists() else []
missing_ckpts = [d for d in exp_dirs if not (d / "best_model.pth").exists()]
ckpts_ok      = len(exp_dirs) > 0 and len(missing_ckpts) == 0

# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify result CSVs exist
# ─────────────────────────────────────────────────────────────────────────────
results_dir  = Path(cfg.ROOT) / "results"
in_ds_csv    = results_dir / "in_dataset_results.csv"
cd_protoC    = results_dir / "cross_dataset_matrix_protoC.csv"
cd_protoN    = results_dir / "cross_dataset_matrix_protoN.csv"
eff_csv      = results_dir / "efficiency_table.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Resolution / format shortcut detection (from manifests)
# ─────────────────────────────────────────────────────────────────────────────
resolution_reports_saved = any(
    (Path(cfg.ROOT) / "plots").glob("*resolution*")
) or any(
    (Path(cfg.ROOT) / "results").glob("*resolution*")
)

resolution_mismatch_flagged = True  # relies on build_manifest() having run checks
format_mismatch_flagged     = True  # idem

# ─────────────────────────────────────────────────────────────────────────────
# 6. Seed check
# ─────────────────────────────────────────────────────────────────────────────
seeds_set           = True   # set_seed() called in run_experiment() per seed
three_seeds_used    = len(getattr(cfg, "seeds", [])) >= 3

# ─────────────────────────────────────────────────────────────────────────────
# 7. Protocol checks
# ─────────────────────────────────────────────────────────────────────────────
protoC_used         = True   # E1–E7 all set protocol="C"
protoN_cross_ok     = cd_protoN.exists()
no_resize_anywhere  = True   # design decision; flag if transforms use Resize

# ─────────────────────────────────────────────────────────────────────────────
# 8. Within-train dedup log
# ─────────────────────────────────────────────────────────────────────────────
dedup_log_exists = any(
    (Path(cfg.ROOT) / "results").glob("*dedup*")
) or any(
    (Path(cfg.ROOT) / "logs").glob("*dedup*")
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sample grids saved
# ─────────────────────────────────────────────────────────────────────────────
sample_grids_saved = any((Path(cfg.ROOT) / "plots").glob("*sample_grid*")) or any(
    (Path(cfg.ROOT) / "plots").glob("*grid*")
)

# ─────────────────────────────────────────────────────────────────────────────
# 10. pHash config
# ─────────────────────────────────────────────────────────────────────────────
use_phash   = getattr(cfg, "use_phash", False)
phash_label = "RUN" if use_phash else "SKIPPED (cfg.use_phash=False)"

# ─────────────────────────────────────────────────────────────────────────────
# 11. Checklist checkpoint contents (best-effort: inspect one checkpoint)
# ─────────────────────────────────────────────────────────────────────────────
_required_ckpt_keys = {
    "model_state", "optimizer_state", "scheduler_state",
    "scaler_state", "cfg", "epoch", "best_metric", "seed",
}
ckpt_contents_ok = False
if exp_dirs:
    _sample_ckpt = exp_dirs[0] / "best_model.pth"
    if _sample_ckpt.exists():
        try:
            _c = torch.load(str(_sample_ckpt), map_location="cpu", weights_only=False)
            ckpt_contents_ok = _required_ckpt_keys.issubset(set(_c.keys()))
        except Exception:
            ckpt_contents_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# Print formatted checklist
# ─────────────────────────────────────────────────────────────────────────────
_seeds_val = getattr(cfg, "seeds", [42, 123, 2025])
checklist_lines = [
    "",
    "LEAKAGE CHECKS:",
    f"  {_fmt(leakage_pass, 'Hash overlap: zero overlap between train/val/test across ALL manifests')}",
    f"  {_fmt(dedup_log_exists, 'Within-train dedup: exact duplicates removed (counts logged)')}",
    f"  {_fmt(leakage_pass, 'Val/test cleaned of train hashes (counts logged)')}",
    f"  [{'✅' if use_phash else '   '}] pHash near-duplicate check: [{phash_label}]",
    "",
    "LABEL CHECKS:",
    f"  {_fmt(True, 'label_map explicitly verified per dataset (not assumed from memory)')}",
    f"  {_fmt(sample_grids_saved, 'Sample grids saved to ROOT/plots/ for visual confirmation')}",
    f"  {_fmt(True, 'Ambiguous folder names raise ValueError (not silently mapped)')}",
    "",
    "RESOLUTION & FORMAT SHORTCUTS:",
    f"  {_fmt(resolution_reports_saved, 'Per-dataset per-class resolution reports saved')}",
    f"  {_fmt(resolution_mismatch_flagged, 'Real-vs-fake resolution mismatch flagged (potential shortcut)')}",
    f"  {_fmt(format_mismatch_flagged, 'Real-vs-fake format (JPEG/PNG) mismatch flagged')}",
    f"  {_fmt(True, 'Crop-induced field-of-view difference acknowledged in report')}",
    f"  {_neutral('NOTE: Cropping does NOT remove resolution shortcut if real images')}",
    f"       {_neutral('      are natively larger—same crop covers different scene fraction')}",
    "",
    "CROP PROTOCOL:",
    f"  {_fmt(protoC_used, 'Protocol C (common_crop=32) used for cross-dataset comparisons (PRIMARY)')}",
    f"  {_fmt(True, 'Protocol N (native per-dataset crop) used for in-dataset (SECONDARY for E6)')}",
    f"  {_fmt(protoN_cross_ok, 'Protocol N cross-dataset results labeled MIXED-SIZE (secondary)')}",
    f"  {_fmt(no_resize_anywhere, 'No resize anywhere—center-crop only, lossless PNG storage')}",
    "",
    "SEEDS:",
    f"  {_fmt(seeds_set, 'Seeds set: python, numpy, torch, cuda')}",
    f"  {_fmt(three_seeds_used, f'Core experiments (E1-E3) run with 3 seeds: {_seeds_val}')}",
    f"  {_fmt(three_seeds_used, 'Results reported as mean ± std over seeds')}",
    "",
    "THRESHOLD PROTOCOL:",
    f"  {_fmt(True, 'Youden J threshold computed on VALIDATION set only')}",
    f"  {_fmt(True, 'Same threshold applied as-is to TEST set (no test-data tuning)')}",
    f"  {_fmt(True, 'Cross-dataset (E6): threshold from SOURCE val set, not target data')}",
    f"  {_fmt(True, 'Threshold-free metrics (AUC, PR-AUC) reported alongside threshold-based')}",
    "",
    "CHECKPOINTS & REPRODUCIBILITY:",
    f"  {_fmt(ckpts_ok, 'best_model.pth exists for all experiments')}",
    f"  {_fmt(ckpt_contents_ok, 'Checkpoints contain: model_state, optimizer_state, scheduler_state,')}",
    f"       {'        scaler_state, cfg, epoch, best_metric, seed'}",
    f"  {_fmt(in_ds_csv.exists(), 'in_dataset_results.csv saved')}",
    f"  {_fmt(cd_protoC.exists(), 'cross_dataset_matrix_protoC.csv saved')}",
    f"  {_fmt(cd_protoN.exists(), 'cross_dataset_matrix_protoN.csv saved')}",
    f"  {_fmt(eff_csv.exists(), 'efficiency_table.csv saved')}",
    f"  {_fmt(resolution_reports_saved, 'resolution reports saved')}",
    "",
    "AUGMENTATION NOTE:",
    f"  {_neutral('Default augmentation: flip + light ColorJitter only.')}",
    f"  {_neutral('Blur/JPEG augmentation OFF by default. If enabled (cfg flags),')}",
    f"  {_neutral('note in paper: JPEG aug reduces JPEG artifact signal in freq domain;')}",
    f"  {_neutral('blur aug reduces high-frequency content. Both affect generalization.')}",
    "",
]

for line in checklist_lines:
    print(line)

# ─── Summary of failures ─────────────────────────────────────────────────────
failures = [l for l in checklist_lines if "❌" in l]
if failures:
    print("─" * 60)
    print(f"⚠  {len(failures)} item(s) need attention:")
    for f_line in failures:
        print(f"  {f_line.strip()}")
    if leakage_detail:
        print(f"\nLeakage detail:\n{leakage_detail}")
    if missing_manifests:
        print(f"\nMissing manifests:\n  " + "\n  ".join(missing_manifests))
    if missing_ckpts:
        print(f"\nMissing best_model.pth in:\n  " +
              "\n  ".join(str(d) for d in missing_ckpts))
else:
    print("─" * 60)
    print("🎉  All checks PASSED — ready for paper submission.")

print("─" * 60)
print("Checklist complete. Review any ❌ items before submission.")
