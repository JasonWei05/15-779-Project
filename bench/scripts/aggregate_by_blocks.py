#!/usr/bin/env python3
"""Aggregate sweep CSV rows by `blocks`.

We average numeric columns for rows that share the same blocks count.
This makes comparison plots cleaner because the sweeps include multiple
factorizations for the same grid size (e.g. 256 blocks).

Usage:
  python3 aggregate_by_blocks.py in.csv out.csv
"""

import csv
import sys


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: aggregate_by_blocks.py in.csv out.csv", file=sys.stderr)
        raise SystemExit(2)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)

    # Identify the blocks column
    try:
        blocks_idx = header.index("blocks")
    except ValueError:
        print("Input CSV missing 'blocks' column", file=sys.stderr)
        raise SystemExit(2)

    # Identify which columns are numeric (based on first non-empty row)
    numeric_cols = set()
    for row in rows:
        if not row:
            continue
        for i, v in enumerate(row):
            if v != "" and is_float(v):
                numeric_cols.add(i)
        break

    groups = {}  # blocks -> {col_idx: [vals], 'first_row': row}
    for row in rows:
        if not row:
            continue
        b = int(float(row[blocks_idx]))
        g = groups.get(b)
        if g is None:
            g = {"first_row": row, "vals": {}}
            groups[b] = g
        for i in numeric_cols:
            v = row[i]
            if v == "":
                continue
            g["vals"].setdefault(i, []).append(float(v))

    out_rows = []
    for b in sorted(groups.keys()):
        g = groups[b]
        base = list(g["first_row"])
        base[blocks_idx] = str(b)
        for i, vals in g["vals"].items():
            if not vals:
                continue
            base[i] = f"{sum(vals)/len(vals):.6f}" if i == blocks_idx else f"{sum(vals)/len(vals):.6f}"
        out_rows.append(base)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in out_rows:
            w.writerow(row)


if __name__ == "__main__":
    main()
