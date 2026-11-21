import re
import numpy as np

tex = open("paper/main.tex", encoding="utf-8").read()

# Strip comments, math, environments, and commands to get pure prose
clean = re.sub(r'%.*$', '', tex, flags=re.MULTILINE)
clean = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', clean, flags=re.DOTALL)
clean = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', clean, flags=re.DOTALL)
clean = re.sub(r'\\begin\{algorithmic\}.*?\\end\{algorithmic\}', '', clean, flags=re.DOTALL)
clean = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', clean, flags=re.DOTALL)
clean = re.sub(r'\\begin\{align\}.*?\\end\{align\}', '', clean, flags=re.DOTALL)
clean = re.sub(r'\$[^$]*\$', '', clean)
clean = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', clean)
clean = re.sub(r'[{}]', '', clean)

# Split into sentences
sentences = re.split(r'[.!?]+', clean)
sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]

lengths = [len(s.split()) for s in sentences]
print(f"Total prose sentences: {len(sentences)}")
print(f"Mean sentence length: {np.mean(lengths):.2f} words")
print(f"Std dev (Burstiness): {np.std(lengths):.2f} (Target > 8.0 for human-like variance)")
print(f"Min length: {min(lengths)} words")
print(f"Max length: {max(lengths)} words")

banned = [
    "delve", "tapestry", "in the ever-evolving", "testament to", "game-changer",
    "pivotal role", "crucial", "furthermore", "moreover", "rapid maturation",
    "democratized", "this paper addresses three open gaps", "the remainder of this paper"
]

found_banned = []
for b in banned:
    if re.search(r'\b' + re.escape(b) + r'\b', tex, re.IGNORECASE):
        found_banned.append(b)

print("Banned AI phrases found:", found_banned or "NONE ✅ (Zero AI tropes)")
