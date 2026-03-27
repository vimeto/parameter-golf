#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "batch59_jobs.json"
INFRA_FAILURES = {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REQUEUED"}
TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}


def ssh_lumi(command: str) -> str:
    proc = subprocess.run(["ssh", "lumi", command], cwd=ROOT, check=True, capture_output=True, text=True)
    return proc.stdout


def job_state(job_id: str) -> tuple[str, str]:
    raw = ssh_lumi(
        f"sacct -X -j {job_id} --parsable2 --noheader --format=JobIDRaw,State,ExitCode | grep '^{job_id}|' | head -n 1"
    ).strip()
    if not raw:
        return "UNKNOWN", ""
    _, state, exit_code = raw.split("|", 2)
    state = state.split()[0]
    state = state.split("+")[0]
    return state, exit_code


def tail_job(job_id: str, lines: int = 40) -> str:
    proc = subprocess.run(
        [str(ROOT / "scripts" / "lumi.sh"), "tail", job_id, str(lines)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def resubmit(entry: dict[str, object]) -> None:
    cmd = list(entry["submit_cmd"])
    proc = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    job_id = proc.stdout.strip().split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse resubmitted job id from: {proc.stdout!r}")
    entry["job_id"] = job_id
    entry["attempt"] = int(entry["attempt"]) + 1


def summarize_tail(text: str) -> str:
    step_matches = re.findall(r"step:(\d+/\d+)", text)
    final_match = re.findall(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    if final_match:
        return f"final_bpb={final_match[-1]}"
    if step_matches:
        return f"step={step_matches[-1]}"
    return "no_step_yet"


def main() -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    poll_s = 180
    max_attempts = 2
    while True:
        pending = 0
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        for entry in registry:
            job_id = str(entry["job_id"])
            state, exit_code = job_state(job_id)
            tail = tail_job(job_id, 30)
            summary = summarize_tail(tail)
            print(f"{entry['run_id']}\tjob={job_id}\tstate={state}\texit={exit_code}\t{summary}")
            if state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "UNKNOWN"}:
                pending += 1
                continue
            if state == "COMPLETED":
                continue
            if state in INFRA_FAILURES and int(entry["attempt"]) < max_attempts:
                print(f"resubmitting infra failure for {entry['run_id']} from job {job_id}")
                resubmit(entry)
                REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
                pending += 1
                continue
            if state in TERMINAL_FAILURES or state in INFRA_FAILURES:
                sys.stderr.write(f"\nTerminal failure in {entry['run_id']} job={job_id} state={state} exit={exit_code}\n")
                sys.stderr.write(tail)
                sys.stderr.write("\n")
                REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
                raise SystemExit(1)
        REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        if pending == 0:
            print("all jobs reached terminal states")
            return
        time.sleep(poll_s)


if __name__ == "__main__":
    main()
