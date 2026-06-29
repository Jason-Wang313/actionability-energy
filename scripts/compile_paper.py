from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=PAPER, check=True)


def run_pdflatex_sequence() -> None:
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    if shutil.which("bibtex"):
        subprocess.run(["bibtex", "main"], cwd=PAPER, check=False)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])


def main() -> int:
    main_tex = PAPER / "main.tex"
    if not main_tex.exists():
        raise SystemExit("paper/main.tex is missing")
    if shutil.which("latexmk"):
        try:
            run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        except subprocess.CalledProcessError:
            if not shutil.which("pdflatex"):
                raise
            run_pdflatex_sequence()
    elif shutil.which("pdflatex"):
        run_pdflatex_sequence()
    else:
        raise SystemExit("No LaTeX compiler found. Install MiKTeX/TeX Live or tectonic.")
    pdf = PAPER / "main.pdf"
    if not pdf.exists() or pdf.stat().st_size < 1000:
        raise SystemExit("LaTeX run did not produce a valid paper/main.pdf")
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
