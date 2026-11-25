"""Run Part 1 from a repository checkout without installing the project package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fdic_ews.cli import main

if __name__ == "__main__":
    main()
