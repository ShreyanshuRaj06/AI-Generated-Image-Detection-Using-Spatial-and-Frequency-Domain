with open("paper/main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines, 1):
    clean = line.split('%')[0] if '%' in line else line
    cnt = clean.count('$')
    if cnt % 2 != 0:
        print(f"Line {idx} ({cnt} dollars): {line.strip()}")
