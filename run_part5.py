"""Develop, evaluate once, or verify saved modelling evidence."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from fdic_ews.modelling.pipeline import develop,evaluate,verify

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['develop','evaluate','verify']); p.add_argument('--workspace',required=True,type=Path); a=p.parse_args()
    root,m={'develop':develop,'evaluate':evaluate,'verify':verify}[a.action](a.workspace)
    print(json.dumps({'status':'PASS','path':str(root),'state':m['state'],'summary':m.get('summary',{}),'rows':m.get('rows')},indent=2))
if __name__=='__main__': main()
