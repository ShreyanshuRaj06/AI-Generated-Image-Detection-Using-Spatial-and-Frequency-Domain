with open("paper/main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(175, 201):
    print(f"{i+1}: {repr(lines[i])}")
