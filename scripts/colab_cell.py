#!/usr/bin/env python3
"""Run specific cells or cell ranges from a local Jupyter notebook on a Colab CLI session."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_NOTEBOOK = "02_adapters_and_variance_test.ipynb"


def parse_cell_specs(specs: list[str], total_cells: int) -> list[int]:
    indices: list[int] = []
    for spec in specs:
        if "-" in spec:
            parts = spec.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            if start < 0 or end >= total_cells or start > end:
                raise ValueError(f"Invalid cell range '{spec}' (valid: 0..{total_cells - 1})")
            indices.extend(range(start, end + 1))
        else:
            idx = int(spec)
            if idx < 0 or idx >= total_cells:
                raise ValueError(f"Cell index {idx} out of range (valid: 0..{total_cells - 1})")
            indices.append(idx)
    return indices


def first_non_empty_line(source: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:90]
    return "(empty)"


def list_cells(nb: dict) -> None:
    for idx, cell in enumerate(nb.get("cells", [])):
        ctype = cell.get("cell_type", "?")
        src = "".join(cell.get("source", []))
        preview = first_non_empty_line(src)
        tag = "CODE" if ctype == "code" else "MD  "
        print(f"[{idx:2d}] {tag} | {preview}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run specific cells from a local .ipynb on a Colab CLI session."
    )
    parser.add_argument(
        "cells",
        nargs="*",
        help="Cell indices or inclusive ranges, e.g. '15' or '1-12' or '1 2 4 15-17'",
    )
    parser.add_argument(
        "-n", "--notebook", default=DEFAULT_NOTEBOOK, help=f"Path to .ipynb (default: {DEFAULT_NOTEBOOK})"
    )
    parser.add_argument("-s", "--session", default="gpu", help="Colab session name (default: gpu)")
    parser.add_argument(
        "-t", "--timeout", default="0", help="colab exec --timeout in seconds (default: 0 = no limit)"
    )
    parser.add_argument("--list", action="store_true", help="List all cells with indices and previews")
    parser.add_argument("--dry-run", action="store_true", help="Print selected cell code without executing")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.is_file():
        print(f"Notebook not found: {nb_path}", file=sys.stderr)
        return 1

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    if args.list:
        list_cells(nb)
        return 0

    if not args.cells:
        parser.print_help()
        return 1

    cells = nb.get("cells", [])
    try:
        selected = parse_cell_specs(args.cells, len(cells))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    for idx in selected:
        cell = cells[idx]
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        if not code.strip():
            continue

        print(f"\n=== Cell {idx} ({first_non_empty_line(code)}) ===", flush=True)
        if args.dry_run:
            print(code)
            continue

        cmd = ["colab", "exec", "-s", args.session, "--timeout", str(args.timeout)]
        res = subprocess.run(cmd, input=code, text=True)
        if res.returncode != 0:
            print(f"Cell {idx} failed with exit code {res.returncode}.", file=sys.stderr)
            return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
