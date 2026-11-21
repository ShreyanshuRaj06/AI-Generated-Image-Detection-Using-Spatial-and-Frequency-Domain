import json

with open("AI_Image_Detection_Colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 10 is the model - extract full source for FusedDetector
cell = nb['cells'][10]
src = "".join(cell.get("source", []))

# Find FusedDetector class
start = src.find("class FusedDetector")
if start >= 0:
    print(src[start:start+5000])
else:
    print("FusedDetector not found in cell 10")
    # search all cells
    for i, c in enumerate(nb['cells']):
        s = "".join(c.get("source", []))
        pos = s.find("class FusedDetector")
        if pos >= 0:
            print(f"Found in cell {i}")
            print(s[pos:pos+5000])
            break
