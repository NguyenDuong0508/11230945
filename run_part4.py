"""Build or independently verify the Gold feature table."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from fdic_ews.features.pipeline import run, resolve_features

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['build','verify']); p.add_argument('--workspace',required=True,type=Path); a=p.parse_args()
    root,m=(run if a.action=='build' else resolve_features)(a.workspace)
    print(json.dumps({'status':'PASS','path':str(root),'summary':m['summary'],'policy':m['policy']},indent=2))
if __name__=='__main__': main()
