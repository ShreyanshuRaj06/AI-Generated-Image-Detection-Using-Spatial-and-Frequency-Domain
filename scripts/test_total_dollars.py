with open("paper/main.tex", "r", encoding="utf-8") as f:
    text = f.read()

total = text.count("$")
print(f"text.count('$') = {total}, is_even = {total % 2 == 0}")
