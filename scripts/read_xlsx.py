import openpyxl
import json

wb = openpyxl.load_workbook(r"C:\Users\LENOVO\.gemini\antigravity\brain\f3a57c64-6170-44ff-8865-97b7c1824af5\.user_uploaded\media_1795292508194_27a00521.xlsx")

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    print(f"\n=== SHEET: {sheet_name} ===")
    print(f"Rows: {ws.max_row}, Cols: {ws.max_column}")
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 60), values_only=False):
        vals = []
        for cell in row:
            v = cell.value
            if v is not None:
                vals.append(str(v)[:200])
            else:
                vals.append("")
        print(" | ".join(vals))
