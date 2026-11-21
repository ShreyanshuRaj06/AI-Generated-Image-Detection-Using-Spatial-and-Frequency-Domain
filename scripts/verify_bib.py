import re

with open("paper/references.bib", "r", encoding="utf-8") as f:
    bib = f.read()

# Match entries
entry_regex = re.compile(r'@(\w+)\s*\{\s*([^,]+),', re.MULTILINE)
entries = list(entry_regex.finditer(bib))

print(f"Total BibTeX entries found: {len(entries)}")

for i in range(len(entries)):
    start = entries[i].start()
    end = entries[i+1].start() if i + 1 < len(entries) else len(bib)
    chunk = bib[start:end].strip()
    key = entries[i].group(2).strip()
    
    # check brace balance
    opens = chunk.count("{")
    closes = chunk.count("}")
    if opens != closes:
        print(f"ERROR in {key}: opens={opens}, closes={closes}")
    else:
        # Check required fields: title, author, year
        if "author" not in chunk.lower():
            print(f"WARNING: missing author in {key}")
        if "title" not in chunk.lower():
            print(f"WARNING: missing title in {key}")
        if "year" not in chunk.lower():
            print(f"WARNING: missing year in {key}")

print("All entries audited successfully.")
