from docx import Document

doc = Document(r"C:\Users\LENOVO\.gemini\antigravity\brain\f3a57c64-6170-44ff-8865-97b7c1824af5\.user_uploaded\media_1795292530410_4154e65d.docx")
for i, para in enumerate(doc.paragraphs):
    if para.text.strip():
        print(f"[P{i}] {para.text}")

print("\n=== TABLES ===")
for ti, table in enumerate(doc.tables):
    print(f"\n--- Table {ti} ---")
    for ri, row in enumerate(table.rows):
        print(" | ".join(c.text.strip() for c in row.cells))
