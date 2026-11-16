import json

with open('colab_downloaded.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Colab notebook: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))[:75].replace('\n', ' ')
    out_types = [o.get('output_type') for o in c.get('outputs', [])]
    has_err = any(t == 'error' for t in out_types)
    status = " [ERR]" if has_err else (" [HAS_OUT]" if out_types else " [NO_OUT]")
    print(f"Cell {i+1:02d} [{ctype[:4]}]{status}: {src}")
