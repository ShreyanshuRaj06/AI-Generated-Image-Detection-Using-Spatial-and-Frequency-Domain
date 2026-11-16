"""
build_notebook.py
=================
Assembles AI_Image_Detection.ipynb from three source files:
  1. cells_1_7.py    — setup, CFG, Kaggle, utilities, DownloadManager
  2. cells_8_12.py   — dataset, model, DCT test, trainer, evaluation
  3. cells_13_to_17.py — experiments, cross-dataset, results, checklist

Each source file uses these markers:
  # [MARKDOWN]      — start of a markdown cell
  # [CODE]          — start of a code cell
  # ===CELL_BREAK=== — boundary between cells

Usage:
  python build_notebook.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUTPUT_NB = Path(__file__).parent / "AI_Image_Detection.ipynb"

# ---------------------------------------------------------------------------
# Source files in order
# ---------------------------------------------------------------------------
SOURCE_FILES = [
    Path(__file__).parent / "cells_1_7.py",
    Path(__file__).parent / "cells_8_12.py",
    Path(__file__).parent / "cells_13_to_17.py",
]

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_source(text: str) -> List[Tuple[str, str]]:
    """
    Split *text* on ``# ===CELL_BREAK===`` boundaries, then determine each
    block's type from the first ``# [MARKDOWN]`` or ``# [CODE]`` marker.

    Returns list of (cell_type, source_text) tuples.
    """
    raw_blocks = re.split(r"\s*#\s*={3,}CELL_BREAK={3,}\s*", text)
    cells: List[Tuple[str, str]] = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Determine type from first marker line
        md_match   = re.match(r"^#\s*\[MARKDOWN\]\s*\n?", block)
        code_match = re.match(r"^#\s*\[CODE\]\s*\n?", block)

        if md_match:
            cell_type = "markdown"
            content   = block[md_match.end():]
        elif code_match:
            cell_type = "code"
            content   = block[code_match.end():]
        else:
            # Fallback: treat as code if it has non-comment content
            stripped = "\n".join(
                l for l in block.splitlines()
                if not l.strip().startswith("#") or l.strip().lstrip("#").strip()
            )
            if stripped.strip():
                cell_type = "code"
                content   = block
            else:
                continue

        # For markdown cells: strip leading "# " from lines (they are
        # inside a Python comment in the source file for syntax highlighting,
        # but should be rendered as plain markdown)
        if cell_type == "markdown":
            md_lines = []
            for line in content.splitlines():
                # Remove a single leading "# " or "#" if present
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            content = "\n".join(md_lines)

        content = content.strip()
        if content:
            cells.append((cell_type, content))

    return cells


# ---------------------------------------------------------------------------
# nbformat 4 helpers
# ---------------------------------------------------------------------------

def _lines(text: str) -> List[str]:
    """Split source into nbformat line list (each line ending with \\n)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def make_code_cell(source: str) -> dict:
    return {
        "cell_type":        "code",
        "execution_count":  None,
        "metadata":         {},
        "outputs":          [],
        "source":           _lines(source),
    }


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata":  {},
        "source":    _lines(source),
    }


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_notebook() -> None:
    nb_cells = []
    total_parsed = 0

    for src_file in SOURCE_FILES:
        if not src_file.exists():
            raise FileNotFoundError(
                f"Source file not found: {src_file}\n"
                "Run the notebook generation cells first."
            )
        text   = src_file.read_text(encoding="utf-8")
        parsed = parse_source(text)
        print(f"  {src_file.name}: {len(parsed)} cells")
        total_parsed += len(parsed)

        for cell_type, source in parsed:
            if cell_type == "markdown":
                nb_cells.append(make_markdown_cell(source))
            else:
                nb_cells.append(make_code_cell(source))

    notebook = {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {
                "name":           "python",
                "version":        "3.10.0",
                "file_extension": ".py",
                "mimetype":       "text/x-python",
                "pygments_lexer": "ipython3",
            },
            "colab": {
                "provenance":     [],
                "gpuType":        "T4",
                "toc_visible":    True,
            },
            "accelerator": "GPU",
        },
        "cells": nb_cells,
    }

    OUTPUT_NB.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    size_kb = OUTPUT_NB.stat().st_size / 1024
    print(f"\n✅ Notebook written → {OUTPUT_NB}")
    print(f"   Cells: {len(nb_cells)} (from {total_parsed} parsed blocks across "
          f"{len(SOURCE_FILES)} source files)")
    print(f"   Size:  {size_kb:.1f} KB")


if __name__ == "__main__":
    print("=" * 60)
    print("Assembling AI_Image_Detection.ipynb")
    print("=" * 60)
    build_notebook()
