import re

with open("paper/main.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Check open/close math dollar signs
dollar_count = 0
for i, c in enumerate(tex):
    if c == '$' and (i == 0 or tex[i-1] != '\\'):
        dollar_count += 1
print(f"Total unescaped dollar signs: {dollar_count} (even: {dollar_count % 2 == 0})")

# Check environments
env_starts = re.findall(r'\\begin\{([^}]+)\}', tex)
env_ends = re.findall(r'\\end\{([^}]+)\}', tex)
print(f"Total begins: {len(env_starts)}, Total ends: {len(env_ends)}")
if env_starts != env_ends:
    from collections import Counter
    c1 = Counter(env_starts)
    c2 = Counter(env_ends)
    print("Diff starts - ends:", c1 - c2)
    print("Diff ends - starts:", c2 - c1)
else:
    print("All environments match perfectly!")

# Check bibtex
with open("paper/references.bib", "r", encoding="utf-8") as f:
    bib = f.read()

entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),', bib)
print(f"\nTotal BibTeX entries: {len(entries)}")
for kind, key in entries:
    # check brace matching for each entry
    sub = bib[bib.find(key):]
    # find next @ or end
    next_at = sub.find("\n@")
    if next_at != -1:
        entry_text = sub[:next_at]
    else:
        entry_text = sub
    b_open = entry_text.count("{")
    b_close = entry_text.count("}")
    if b_open != b_close:
        print(f"Warning: Brace imbalance in entry {key}: open={b_open}, close={b_close}")

print("BibTeX check complete.")
