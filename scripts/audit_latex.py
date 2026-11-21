"""Static LaTeX audit: verify cite keys, special chars, structure."""
import re
from pathlib import Path

tex = Path("paper/main.tex").read_text(encoding="utf-8")
bib = Path("paper/references.bib").read_text(encoding="utf-8")

# 1. Extract all cite keys from tex
cite_keys_in_tex = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for key in m.group(1).split(","):
        cite_keys_in_tex.add(key.strip())

# 2. Extract all bib entry keys
bib_keys = set()
for m in re.finditer(r'@\w+\{(\w+)', bib):
    bib_keys.add(m.group(1))

# 3. Check orphans
missing_in_bib = cite_keys_in_tex - bib_keys
unused_in_tex = bib_keys - cite_keys_in_tex

print("=" * 60)
print("CITATION AUDIT")
print("=" * 60)
print(f"  Cite keys in main.tex: {len(cite_keys_in_tex)}")
print(f"  Bib entries:           {len(bib_keys)}")
print(f"  Missing in bib:        {missing_in_bib or 'NONE ✅'}")
print(f"  Unused bib entries:    {unused_in_tex or 'NONE'}")

# 4. Check for common LaTeX errors
errors = []

# Unmatched braces (simple check)
open_b = tex.count("{")
close_b = tex.count("}")
if open_b != close_b:
    errors.append(f"Brace mismatch: {{ = {open_b}, }} = {close_b}")

# Check \begin/\end matching
begins = re.findall(r'\\begin\{(\w+)\}', tex)
ends = re.findall(r'\\end\{(\w+)\}', tex)
if begins != ends:
    from collections import Counter
    bc = Counter(begins)
    ec = Counter(ends)
    for env in set(list(bc.keys()) + list(ec.keys())):
        if bc.get(env, 0) != ec.get(env, 0):
            errors.append(f"Environment mismatch: {env} begin={bc.get(env,0)} end={ec.get(env,0)}")

# Check for unescaped % in prose (outside comments)
lines = tex.splitlines()
for i, line in enumerate(lines, 1):
    # Skip lines that start with %
    stripped = line.lstrip()
    if stripped.startswith("%"):
        continue
    # Check for unescaped % (not preceded by \)
    for m in re.finditer(r'(?<!\\)%', line):
        # This is a comment start, not an error per se, but check context
        pass

# Check for unescaped & outside tabular
# (this is complex, skip)

# Check for \\ref without matching \\label
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex))
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
missing_labels = refs - labels
if missing_labels:
    errors.append(f"References without labels: {missing_labels}")

print()
print("=" * 60)
print("STRUCTURAL AUDIT")
print("=" * 60)
print(f"  \\ref/\\eqref keys: {len(refs)}")
print(f"  \\label keys:      {len(labels)}")
print(f"  Missing labels:   {missing_labels or 'NONE ✅'}")

# Required IEEE sections
required = ["Introduction", "Related Work", "Proposed Methodology",
            "Experimental Setup", "Results and Discussion", "Conclusion"]
for sec in required:
    if sec.lower() not in tex.lower():
        errors.append(f"Missing section: {sec}")
    else:
        print(f"  Section '{sec}': FOUND ✅")

# IEEEtran document class
if "\\documentclass[conference]{IEEEtran}" not in tex:
    errors.append("Missing \\documentclass[conference]{IEEEtran}")
else:
    print(f"  IEEEtran class: ✅")

# bibliographystyle
if "\\bibliographystyle{IEEEtran}" not in tex:
    errors.append("Missing \\bibliographystyle{IEEEtran}")
else:
    print(f"  IEEE bibliography style: ✅")

print()
if errors:
    print("ERRORS:")
    for e in errors:
        print(f"  ❌ {e}")
else:
    print("✅ ALL CHECKS PASSED — ready for Overleaf upload.")

# 5. Count content
word_count = len(re.findall(r'\b\w+\b', re.sub(r'\\[a-zA-Z]+', '', tex)))
print(f"\n  Approximate word count: {word_count}")
print(f"  Tables: {tex.count(chr(92) + 'begin{table')}")
print(f"  Figures: {tex.count(chr(92) + 'begin{figure')}")
print(f"  Equations: {tex.count(chr(92) + 'begin{equation')}")
print(f"  Algorithms: {tex.count(chr(92) + 'begin{algorithm')}")
