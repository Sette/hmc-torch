#!/usr/bin/env python3
"""Merge `runs` from a staging protocol directory into the main protocol JSONs.

Usage:
    python3 docs/specs/merge_runs.py <staging_dir> <dest_dir>

Examples:
    # consolidate GO bce/mc into output/protocol/<dataset>.json
    python3 docs/specs/merge_runs.py output/protocol_go_recipe/bce output/protocol
    python3 docs/specs/merge_runs.py output/protocol_go_recipe/mc  output/protocol

    # merge hinge runs
    python3 docs/specs/merge_runs.py output/protocol_hinge output/protocol

Runs are deduplicated by (objective, seed); existing runs are preserved.
"""
import glob
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    staging, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)
    total = 0
    for src in sorted(glob.glob(os.path.join(staging, "*.json"))):
        name = os.path.basename(src)
        dst = os.path.join(dest, name)
        with open(src) as f:
            incoming = json.load(f)
        if os.path.exists(dst):
            with open(dst) as f:
                record = json.load(f)
        else:
            record = json.loads(json.dumps(incoming))
            record["runs"] = []
        seen = {(r.get("objective"), r.get("seed")) for r in record.get("runs", [])}
        added = 0
        for run in incoming.get("runs", []):
            key = (run.get("objective"), run.get("seed"))
            if key not in seen:
                record["runs"].append(run)
                seen.add(key)
                added += 1
        with open(dst, "w") as f:
            json.dump(record, f, indent=1, default=float)
        print(f"[merge] {name}: {added} run(s) added")
        total += added
    print(f"[merge] done: {total} run(s) added from {staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
