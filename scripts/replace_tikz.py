import re

with open("paper/main.tex", "r", encoding="utf-8") as f:
    content = f.read()

# Replace Figure 1
fig_pattern = re.compile(r'\\begin\{figure\*\}\[!t\].*?\\label\{fig:architecture\}\s*\\end\{figure\*\}', re.DOTALL)
assert fig_pattern.search(content), "Figure 1 pattern not found!"

from test_perfect_tikz import test_tikz

content = fig_pattern.sub(lambda m: test_tikz, content)

with open("paper/main.tex", "w", encoding="utf-8") as f:
    f.write(content)

print("Replaced Figure 1 in main.tex successfully!")
