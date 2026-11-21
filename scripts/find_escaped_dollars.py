with open("paper/main.tex", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'\\\$', text)]
print(f"Number of escaped dollars \\$: {len(matches)}")
for idx in matches:
    print(f"Context: {text[max(0, idx-20):idx+20]}")
