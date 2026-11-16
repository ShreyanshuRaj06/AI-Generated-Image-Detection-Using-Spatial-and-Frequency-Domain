"""Quick structural validation of the assembled notebook."""
import json
from pathlib import Path

nb_path = Path(r"C:\Users\LENOVO\Desktop\RMIPR\AI_Image_Detection.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

fmt = nb["nbformat"]
fmt_min = nb["nbformat_minor"]
cells = nb["cells"]
md_count   = sum(1 for c in cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in cells if c["cell_type"] == "code")

print(f"nbformat      : {fmt}.{fmt_min}")
print(f"Total cells   : {len(cells)}")
print(f"  Markdown    : {md_count}")
print(f"  Code        : {code_count}")
print(f"File size     : {nb_path.stat().st_size / 1024:.1f} KB")
print()

# Key checks
checks = {
    "CFG dataclass defined": any("class CFG" in "".join(c["source"]) for c in cells),
    "set_seed defined": any("def set_seed" in "".join(c["source"]) for c in cells),
    "DownloadManager defined": any("class DownloadManager" in "".join(c["source"]) for c in cells),
    "AIImageDataset defined": any("class AIImageDataset" in "".join(c["source"]) for c in cells),
    "SpatialBranch defined": any("class SpatialBranch" in "".join(c["source"]) for c in cells),
    "FrequencyBranch defined": any("class FrequencyBranch" in "".join(c["source"]) for c in cells),
    "FusedDetector defined": any("class FusedDetector" in "".join(c["source"]) for c in cells),
    "DCT unit test present": any("test_dct_implementation" in "".join(c["source"]) for c in cells),
    "Hann window test present": any("test_hann_window" in "".join(c["source"]) for c in cells),
    "Trainer defined": any("class Trainer" in "".join(c["source"]) for c in cells),
    "evaluate_test_set defined": any("def evaluate_test_set" in "".join(c["source"]) for c in cells),
    "Youden threshold present": any("compute_youden_threshold" in "".join(c["source"]) for c in cells),
    "assert_no_leakage present": any("def assert_no_leakage" in "".join(c["source"]) for c in cells),
    "run_experiment present": any("def run_experiment" in "".join(c["source"]) for c in cells),
    "run_cross_dataset_experiment present": any("def run_cross_dataset_experiment" in "".join(c["source"]) for c in cells),
    "Protocol C/N handling present": any("Protocol C" in "".join(c["source"]) or "protocol" in "".join(c["source"]) for c in cells),
    "E6 cross-dataset present": any("E6" in "".join(c["source"]) for c in cells),
    "Checklist cell present": any("Pre-Publication" in "".join(c["source"]) for c in cells),
    "No resize anywhere": not any("transforms.Resize" in "".join(c["source"]) or "T.Resize" in "".join(c["source"]) for c in cells),
    "BCEWithLogitsLoss used": any("BCEWithLogitsLoss" in "".join(c["source"]) for c in cells),
    "AMP (autocast) used": any("autocast" in "".join(c["source"]) for c in cells),
    "Leakage check: sha256 overlap": any("sha256" in "".join(c["source"]) and "overlap" in "".join(c["source"]) for c in cells),
}

all_pass = True
for label, ok in checks.items():
    sym = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{sym}] {label}")

print()
print("Cell index / first 90 chars of source:")
for i, c in enumerate(cells):
    src = "".join(c["source"]).strip().replace("\n", " ")[:90]
    print(f"  {i+1:02d} [{c['cell_type']:8s}]: {src}")

print()
if all_pass:
    print("ALL CHECKS PASSED — notebook is structurally complete.")
else:
    print("One or more checks FAILED — review above.")
