with open("paper/main.tex", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()

for i, line in enumerate(lines, 1):
    cnt = line.count("$")
    if cnt % 2 != 0:
        print(f"Line {i} has {cnt} dollar signs: {line}")
