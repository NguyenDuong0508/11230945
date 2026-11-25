"""Part 1 command-line workflow."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

from .acquisition import acquire, publish_source, resolve_pointer, run_id, source_run, verify
from .benchmark import run
from .contracts import IntegrityError, sha256, write_json, read_json
from .storage import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    a = sub.add_parser("acquire"); a.add_argument("--workspace", type=Path, required=True)
    a.add_argument("--fresh", action="store_true"); a.add_argument("--max-new-quarters", type=int)
    v = sub.add_parser("verify"); v.add_argument("--workspace", type=Path, required=True)
    b = sub.add_parser("benchmark"); b.add_argument("--workspace", type=Path, required=True)
    b.add_argument("--local", type=Path, required=True); b.add_argument("--warmups", type=int, default=1)
    b.add_argument("--repetitions", type=int, default=5); b.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    ws = args.workspace.resolve(); ws.mkdir(parents=True, exist_ok=True)
    if args.action == "acquire":
        root = source_run(ws, fresh=args.fresh)
        result = acquire(root, max_new_quarters=args.max_new_quarters)
        if result["status"] == "PASS":
            publish_source(ws, root, result)
        print(json.dumps({k: v for k, v in result.items() if k != "quarter_statistics"}, indent=2))
    elif args.action == "verify":
        result = verify(resolve_pointer(ws))
        print(json.dumps({k: v for k, v in result.items() if k != "quarter_statistics"}, indent=2))
    else:
        if (ws / "pending_source.json").exists():
            raise IntegrityError("Complete the pending acquisition before benchmarking")
        source = resolve_pointer(ws)
        ident = run_id()
        local = args.local.resolve() / ident
        if local == ws or ws in local.parents or local in ws.parents:
            raise IntegrityError("Benchmark scratch directory must be separate from the workspace")
        local.mkdir(parents=True, exist_ok=False)
        work, evidence = local / "formats", local / "evidence"
        prepare(source, work)
        result = run(work, evidence, args.warmups, args.repetitions, args.threads)
        if result["source_manifest_sha256"] != sha256(source / "manifest.json"):
            raise IntegrityError("Source changed during benchmark")
        target = ws / "storage_evidence" / ident
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(evidence, target)
        for name, digest in result["files"].items():
            if sha256(target / name) != digest:
                raise IntegrityError("Evidence copy differs from worker result")
        write_json(ws / "active_storage.json", {"path": target.relative_to(ws).as_posix(),
                   "source_manifest_sha256": result["source_manifest_sha256"], "evidence_sha256": sha256(target / "evidence.json")})
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        # Delete only the temporary, successfully verified local benchmark run.
        shutil.rmtree(local)


if __name__ == "__main__":
    main()
