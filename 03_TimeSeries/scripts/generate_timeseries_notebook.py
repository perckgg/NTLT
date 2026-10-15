"""
Generate the Jupyter notebook notebooks/03_TimeSeries_Air_Quality.ipynb
Time Series part: 24-hour-ahead PM2.5 forecasting — Hanoi vs Ho Chi Minh City.
Follows the 4-section structure (Section 5) and the rubric of the project brief v4.

Run from the 03_TimeSeries folder:  py scripts/generate_timeseries_notebook.py
Cell contents live in scripts/nb_parts/*.py; observations live in scripts/nb_parts/notes.py.
"""

import json
import os
import textwrap

OUTPUT_PATH = os.path.join("notebooks", "03_TimeSeries_Air_Quality.ipynb")
PARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb_parts")
PART_FILES = ["part1_overview.py", "part2_preparation.py", "part3_modeling.py", "part4_summary.py"]

cells = []


def _lines(text):
    text = textwrap.dedent(text).strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(text)})


def load_notes():
    ns = {}
    path = os.path.join(PARTS_DIR, "notes.py")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            exec(fh.read(), ns)
    return ns.get("NOTES", {})


def build():
    notes = load_notes()
    for name in PART_FILES:
        with open(os.path.join(PARTS_DIR, name), encoding="utf-8") as fh:
            exec(fh.read(), {"md": md, "code": code,
                             "note": lambda k: notes.get(k, "_(To be updated after running the notebook)_")})
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": cells,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"Created {OUTPUT_PATH} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
