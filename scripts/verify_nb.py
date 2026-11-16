import json

with open("AI_Image_Detection_Colab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
last_cell = nb['cells'][-1]
print("Last cell outputs:")
for o in last_cell.get("outputs", []):
    text = "".join(o.get("text", []))
    print(text)
