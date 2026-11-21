import json

with open("AI_Image_Detection_Colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 2 = CFG (index 2)
# Cell 10 = Model Architecture (index 10)
# Cell 14 = Trainer (index 14)

for ci in [2, 10, 14, 16, 18]:
    cell = nb['cells'][ci]
    src = "".join(cell.get("source", []))
    print(f"\n{'='*80}")
    print(f"CELL {ci} SOURCE ({len(src)} chars, {len(src.splitlines())} lines)")
    print(f"{'='*80}")
    print(src[:8000])
    if len(src) > 8000:
        print(f"\n... [TRUNCATED, total {len(src)} chars] ...")
