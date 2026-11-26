"""Print the cross-stage evidence synthesis without writing workspace artifacts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from fdic_ews.conclusions import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.workspace), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
