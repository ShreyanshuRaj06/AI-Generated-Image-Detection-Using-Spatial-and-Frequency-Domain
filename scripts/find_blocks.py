# Let's inspect the exact lines of Table I and Figure 1 in main.tex
with open("paper/main.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Table I starts around line 135-155:")
for i in range(130, 160):
    if "\\begin{table" in lines[i]:
        print(f"Table I start at line {i+1}")
    if "\\end{table" in lines[i]:
        print(f"Table I end at line {i+1}")

print("Figure 1 starts around line 165-225:")
for i in range(160, 230):
    if "\\begin{figure" in lines[i]:
        print(f"Fig 1 start at line {i+1}")
    if "\\end{figure" in lines[i]:
        print(f"Fig 1 end at line {i+1}")
