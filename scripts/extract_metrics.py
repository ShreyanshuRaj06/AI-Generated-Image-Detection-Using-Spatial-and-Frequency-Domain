import json

with open("AI_Image_Detection_Colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    ct = cell["cell_type"]
    outputs = cell.get("outputs", [])

    # Extract first line for header
    first_line = src.split("\n")[0][:120] if src else ""
    out_text = ""
    for o in outputs:
        t = "".join(o.get("text", []))
        out_text += t

    # Print cell info
    print(f"=== CELL {i} [{ct}] ===")
    print(f"  HEADER: {first_line}")
    print(f"  SRC LINES: {len(src.splitlines())}")
    print(f"  OUTPUT CHARS: {len(out_text)}")

    # Print key outputs (metrics, results, configs)
    if out_text:
        # Look for key patterns
        for line in out_text.splitlines():
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['auc', 'accuracy', 'f1', 'precision', 'recall',
                                                 'loss', 'epoch', 'threshold', 'roc',
                                                 'cfg', 'config', 'lr=', 'batch',
                                                 'params', 'flops', 'latency',
                                                 'cross_dataset', 'protocol',
                                                 'gravex', 'faces', 'cifake', 'cashbowman',
                                                 'spatial', 'frequency', 'fused',
                                                 'best_', 'checklist', 'pass',
                                                 'resolution', 'train=', 'val=', 'test=',
                                                 'overlap', 'hash', 'dedup']):
                print(f"    >> {line.strip()[:200]}")
    print()
