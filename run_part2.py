"""Build or verify the Part 2 Silver panel."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from fdic_ews.quality.panel import run, resolve_panel

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("build","verify"))
    parser.add_argument("--workspace",required=True,type=Path)
    args=parser.parse_args()
    path,manifest=(run if args.action=="build" else resolve_panel)(args.workspace)
    print(json.dumps({"status":"PASS","path":str(path),"summary":manifest["summary"]},indent=2))
