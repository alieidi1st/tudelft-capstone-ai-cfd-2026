"""Convert a cell-marked Python script into a Jupyter notebook.

Cell markers, each on its own line:

    # %% [markdown]        -> a markdown cell (following ``#`` comments are the text)
    # %%                   -> a code cell

Usage:
    python tools/py_to_notebook.py input.py output.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf


def convert(src: Path, dst: Path, kernel_name: str = "python3") -> None:
    lines = src.read_text().splitlines()

    cells, kind, buf = [], None, []

    def flush() -> None:
        if kind is None:
            return
        if kind == "markdown":
            text = "\n".join(
                ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                for ln in buf
            ).strip("\n")
            if text.strip():
                cells.append(nbf.v4.new_markdown_cell(text))
        else:
            code = "\n".join(buf).strip("\n")
            if code.strip():
                cells.append(nbf.v4.new_code_cell(code))

    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        else:
            buf.append(ln)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": kernel_name},
        "language_info": {"name": "python"},
    })
    dst.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(dst))
    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    n_code = len(cells) - n_md
    print(f"{dst}: {len(cells)} cells ({n_md} markdown, {n_code} code)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
