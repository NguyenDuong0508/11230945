"""Command-line entry point for the scalable processing protocol."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from fdic_ews.processing.benchmark import run,verify_evidence

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','verify']); p.add_argument('--workspace',type=Path,required=True); p.add_argument('--scratch',type=Path,default=Path('/tmp/fdic_processing')); a=p.parse_args()
    if a.action=='run': print('Verified processing evidence:',run(a.workspace,a.scratch))
    else:
        path,m=verify_evidence(a.workspace); print('Processing evidence: PASS',path); print('Measured trials:',m['measured_trials'])
