with open("paper/main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

in_math = False
for idx, line in enumerate(lines, 1):
    # remove comments
    clean = ""
    i = 0
    while i < len(line):
        if line[i] == '%' and (i == 0 or line[i-1] != '\\'):
            break
        clean += line[i]
        i += 1
    
    # count unescaped $
    j = 0
    dollars_on_line = []
    while j < len(clean):
        if clean[j] == '$' and (j == 0 or clean[j-1] != '\\'):
            dollars_on_line.append(j)
            in_math = not in_math
        j += 1
    
    if len(dollars_on_line) % 2 != 0:
        print(f"Line {idx} has ODD dollars ({len(dollars_on_line)}): {line.strip()}")
        print(f"  --> now in_math = {in_math}")
