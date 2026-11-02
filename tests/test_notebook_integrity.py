"""
tests/test_notebook_integrity.py
--------------------------------
Validates that all Colab research notebooks (01 through 09):
1. Are valid, well-formed JSON documents.
2. Have valid Python syntax across all code cells (after filtering IPython magics).
3. Contain no broken backslash continuations or unclosed strings.
"""

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "colab"
LOW_LABEL_NOTEBOOKS_DIR = REPO_ROOT / "notebooks" / "low_label"


def _validate_notebooks_in_dir(directory: Path, min_count: int):
    notebook_files = sorted(directory.glob("*.ipynb"))
    assert len(notebook_files) >= min_count, f"Expected at least {min_count} notebooks in {directory}, found {len(notebook_files)}"

    for nb_path in notebook_files:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert "cells" in nb, f"Notebook {nb_path.name} is missing 'cells' key."
        cells = nb["cells"]
        assert len(cells) > 0, f"Notebook {nb_path.name} contains no cells."

        for idx, cell in enumerate(cells):
            if cell.get("cell_type") == "code":
                source_lines = cell.get("source", [])
                py_lines = []
                in_bash_continuation = False

                for line in source_lines:
                    stripped = line.strip()
                    if stripped.startswith(("%", "!")):
                        in_bash_continuation = stripped.endswith("\\")
                        continue
                    if in_bash_continuation:
                        in_bash_continuation = stripped.endswith("\\")
                        continue
                    py_lines.append(line)

                py_code = "".join(py_lines)
                if py_code.strip():
                    try:
                        ast.parse(py_code)
                    except SyntaxError as e:
                        raise AssertionError(
                            f"Syntax error in {nb_path.name} cell {idx}: {e}\nCode:\n{py_code}"
                        ) from e


def test_colab_notebooks_json_and_ast_integrity():
    """Validates JSON structure and Python AST syntax of all Colab downstream notebooks."""
    _validate_notebooks_in_dir(NOTEBOOKS_DIR, min_count=9)


def test_low_label_notebooks_json_and_ast_integrity():
    """Validates JSON structure and Python AST syntax of all 8 low-label parallel notebooks."""
    _validate_notebooks_in_dir(LOW_LABEL_NOTEBOOKS_DIR, min_count=8)


if __name__ == "__main__":
    test_colab_notebooks_json_and_ast_integrity()
    test_low_label_notebooks_json_and_ast_integrity()
    print("✓ All Colab and low-label notebooks passed JSON and Python AST integrity checks!")
