"""
Fix stock CSV data in-place.
- Keeps all 1830 rows per stock (no deletions)
- Leaves Close prices untouched
- Adjusts High = max(O,H,L,C) and Low = min(O,H,L,C) so OHLC invariant holds
"""

import os
import csv

DATA_DIR = r"InvestingCom_EGX70_6Years_Aligned"

def fix_stock(filepath):
    rows_fixed = 0
    
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_rows = list(reader)
    
    for row in all_rows:
        if len(row) < 6 or all(c.strip() == '' for c in row):
            continue
        try:
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
        except (ValueError, IndexError):
            continue
        
        new_h = max(o, h, l, c)
        new_l = min(o, h, l, c)
        
        if new_h != h or new_l != l:
            rows_fixed += 1
            row[2] = f"{new_h:.14f}"
            row[3] = f"{new_l:.14f}"
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)
    
    return rows_fixed

def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Fixing {len(files)} files in-place in {DATA_DIR}/")
    print()
    
    total = 0
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        fixed = fix_stock(path)
        total += fixed
        status = f"fixed {fixed} rows" if fixed > 0 else "OK"
        print(f"  {fname}: {status}")
    
    print(f"\nTotal rows fixed: {total}")

if __name__ == '__main__':
    main()
