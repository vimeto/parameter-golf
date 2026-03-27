#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "specs" / "batch59" / "full_node_runs.tsv"
REGISTRY = ROOT / "experiments" / "batch59_jobs.json"
TRAIN_SCRIPT = "specs/batch59/run_research_backbone.py"
SBATCH_TIME = "01:45:00"
PARTITION = "dev-g"


def load_runs() -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        run_id, env_str = line.split("\t", 1)
        runs.append({"run_id": run_id, "env_str": env_str})
    return runs


def build_submit_cmd(run: dict[str, str]) -> list[str]:
    env_parts = ["ALL", "NUM_GPUS=8", f"TRAIN_SCRIPT={TRAIN_SCRIPT}"] + shlex.split(run["env_str"])
    return [
        str(ROOT / "scripts" / "lumi.sh"),
        "submit",
        "--parsable",
        "--partition",
        PARTITION,
        "--job-name",
        f"pg59-{run['run_id']}",
        "--time",
        SBATCH_TIME,
        "--export",
        ",".join(env_parts),
        "scripts/slurm/lumi_train.sh",
    ]


def submit(run: dict[str, str]) -> dict[str, object]:
    cmd = build_submit_cmd(run)
    proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    job_id = proc.stdout.strip().split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse job id from: {proc.stdout!r}")
    return {
        "run_id": run["run_id"],
        "job_id": job_id,
        "attempt": 1,
        "env_str": run["env_str"],
        "submit_cmd": cmd,
    }


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if dry_run:
        for run in runs:
            print(" ".join(shlex.quote(part) for part in build_submit_cmd(run)))
        return
    registry = [submit(run) for run in runs]
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Wrote {REGISTRY}")
    for entry in registry:
        print(f"{entry['run_id']}\t{entry['job_id']}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        raise
