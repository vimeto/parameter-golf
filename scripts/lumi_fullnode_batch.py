from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "experiments" / "batch59_jobs.json"
TRAIN_LAUNCHER = "scripts/slurm/lumi_train.sh"

NODE_FAILURE_STATES = {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "REVOKED"}
SUCCESS_STATES = {"COMPLETED"}
TERMINAL_FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "DEADLINE",
}
FAIL_FAST_PATTERNS = {
    "traceback": re.compile(r"Traceback \\(most recent call last\\):"),
    "runtime_error": re.compile(r"RuntimeError:"),
    "death_signal": re.compile(r"death signal"),
    "rccl_error": re.compile(r"RCCL.*error", re.IGNORECASE),
    "miopen_error": re.compile(r"MIOpen.*error", re.IGNORECASE),
    "hip_error": re.compile(r"HIP error|hipError", re.IGNORECASE),
    "rocblas_error": re.compile(r"rocBLAS.*error", re.IGNORECASE),
}


@dataclass
class JobRun:
    env: dict[str, str]
    run_id: str
    attempt: int = 0
    job_id: str | None = None
    last_state: str | None = None
    feature_verified: bool = False
    last_step: str | None = None
    stalled_polls: int = 0
    history: list[dict[str, str]] = field(default_factory=list)


def run_local(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def run_remote(command: str) -> str:
    proc = subprocess.run(["ssh", "lumi", command], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Remote command failed: {command}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def parse_manifest(path: Path) -> list[JobRun]:
    runs: list[JobRun] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        env: dict[str, str] = {}
        for token in shlex.split(line):
            key, value = token.split("=", 1)
            env[key] = value
        run_id = env.get("RUN_ID")
        if not run_id:
            raise ValueError(f"Manifest line is missing RUN_ID: {line}")
        runs.append(JobRun(env=env, run_id=run_id))
    if not runs:
        raise ValueError(f"No runnable lines found in {path}")
    return runs


def load_state(path: Path) -> list[JobRun]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [JobRun(**entry) for entry in payload]


def write_state(path: Path, runs: list[JobRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(run) for run in runs]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def submit_run(run: JobRun, args: argparse.Namespace) -> None:
    export_items = {
        "NUM_GPUS": "8",
        "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
        "STRICT_FEATURE_CHECK": "1",
        "TRAIN_SCRIPT": args.train_script,
        "TRAIN_LOG_EVERY": str(args.train_log_every),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
    }
    export_items.update(run.env)
    export_arg = "--export=ALL," + ",".join(f"{k}={v}" for k, v in export_items.items())
    job_name = f"{args.job_name_prefix}-{run.run_id[:40]}"
    out = run_local(
        [
            "./scripts/lumi.sh",
            "submit",
            f"--time={args.time_limit}",
            f"--job-name={job_name}",
            export_arg,
            TRAIN_LAUNCHER,
        ]
    )
    match = re.search(r"Submitted batch job (\d+)", out)
    if not match:
        raise RuntimeError(f"Could not parse sbatch output for {run.run_id}: {out}")
    run.job_id = match.group(1)
    run.attempt += 1
    run.last_state = "SUBMITTED"
    run.feature_verified = False
    run.last_step = None
    run.stalled_polls = 0
    run.history.append({"job_id": run.job_id, "event": "submitted", "attempt": str(run.attempt)})
    print(f"[submit] {run.run_id} -> job {run.job_id} (attempt {run.attempt})")


def query_state(job_id: str) -> str:
    squeue_state = run_remote(f"squeue -h -j {job_id} -o '%T' | head -n 1")
    if squeue_state:
        return squeue_state.strip().split()[0]
    sacct = run_remote(f"sacct -X -P -n -j {job_id} --format=JobIDRaw,State,ExitCode,Elapsed | head -n 20")
    for line in sacct.splitlines():
        parts = line.split("|")
        if parts and parts[0] == job_id:
            return parts[1].split()[0]
    return "UNKNOWN"


def tail_logs(job_id: str, lines: int = 30) -> str:
    stdout, stderr = tail_log_parts(job_id, lines=lines)
    return f"{stdout}\n---ERR---\n{stderr}".strip()


def tail_log_parts(job_id: str, lines: int = 30) -> tuple[str, str]:
    stdout = run_remote(f"tail -n {lines} ~/parameter-golf/logs/pgolf_{job_id}.out 2>/dev/null")
    stderr = run_remote(f"tail -n {lines} ~/parameter-golf/logs/pgolf_{job_id}.err 2>/dev/null")
    return stdout, stderr


def extract_latest_step(text: str) -> str | None:
    matches = re.findall(r"step:(\d+/\d+)", text)
    return matches[-1] if matches else None


def find_fail_fast_marker(text: str) -> str | None:
    for name, pattern in FAIL_FAST_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def confirm_feature_logs(run: JobRun) -> None:
    assert run.job_id is not None
    grep_cmd = (
        "grep -E 'FEATURE VERIFICATION|research_family|hnet_active|state_mixer_active|ut_active|"
        "random_adapter_active|megakernel_compiled_loss|extra_feature_params' "
        f"~/parameter-golf/logs/pgolf_{run.job_id}.out 2>/dev/null | tail -n 20"
    )
    out = run_remote(grep_cmd)
    if out:
        print(f"[verify] {run.run_id} job {run.job_id}\n{out}")
        run.feature_verified = True


def parse_results(job_id: str) -> dict[str, str]:
    log = run_remote(f"cat ~/parameter-golf/logs/pgolf_{job_id}.out 2>/dev/null")
    result: dict[str, str] = {}
    for pattern, key in [
        (r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", "val_bpb"),
        (r"Total submission size int8\\+zlib: ([0-9]+) bytes", "artifact_bytes"),
        (r"step:([0-9]+)/", "steps"),
        (r"peak memory allocated: ([0-9]+) MiB", "peak_mem_mib"),
    ]:
        match = re.search(pattern, log)
        if match:
            result[key] = match.group(1)
    return result


def monitor_runs(runs: list[JobRun], args: argparse.Namespace) -> None:
    while True:
        unfinished = [run for run in runs if run.last_state not in SUCCESS_STATES | TERMINAL_FAILURE_STATES]
        if not unfinished:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[poll {timestamp}] checking {len(unfinished)} jobs")
        for run in unfinished:
            assert run.job_id is not None
            state = query_state(run.job_id)
            if state != run.last_state:
                print(f"[state] {run.run_id} job {run.job_id}: {run.last_state} -> {state}")
                run.history.append({"job_id": run.job_id, "event": "state", "state": state})
                run.last_state = state
            stdout_tail, stderr_tail = tail_log_parts(run.job_id, lines=80)
            combined_tail = f"{stdout_tail}\n{stderr_tail}"
            marker = find_fail_fast_marker(combined_tail)
            if marker:
                run_local(["./scripts/lumi.sh", "cancel", run.job_id])
                raise RuntimeError(f"{run.run_id} hit fail-fast marker {marker}\n{combined_tail}")
            if state == "RUNNING":
                if not run.feature_verified:
                    confirm_feature_logs(run)
                latest_step = extract_latest_step(stdout_tail)
                if latest_step and latest_step == run.last_step:
                    run.stalled_polls += 1
                elif latest_step:
                    run.last_step = latest_step
                    run.stalled_polls = 0
                if args.stall_polls > 0 and run.stalled_polls >= args.stall_polls and run.last_step is not None:
                    run_local(["./scripts/lumi.sh", "cancel", run.job_id])
                    raise RuntimeError(f"{run.run_id} stalled at step {run.last_step}\n{combined_tail}")
            if state in NODE_FAILURE_STATES:
                if run.attempt <= args.max_node_retries:
                    print(f"[retry] {run.run_id} job {run.job_id} ended with {state}, resubmitting")
                    run.history.append({"job_id": run.job_id, "event": "node_retry", "state": state})
                    submit_run(run, args)
                else:
                    logs = tail_logs(run.job_id)
                    raise RuntimeError(f"{run.run_id} exceeded node-failure retries with state {state}\n{logs}")
            elif state in TERMINAL_FAILURE_STATES:
                logs = tail_logs(run.job_id)
                raise RuntimeError(f"{run.run_id} failed with state {state}\n{logs}")
            elif state in SUCCESS_STATES:
                summary = parse_results(run.job_id)
                summary_text = " ".join(f"{k}={v}" for k, v in summary.items()) if summary else "no_summary_found"
                print(f"[done] {run.run_id} job {run.job_id} {summary_text}")
        write_state(args.state_path, runs)
        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and monitor full-node LUMI research runs.")
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--train-script", default="specs/batch59/run_research_families.py")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--time-limit", default="01:30:00")
    parser.add_argument("--max-wallclock-seconds", type=int, default=3000)
    parser.add_argument("--max-node-retries", type=int, default=1)
    parser.add_argument("--job-name-prefix", default="pg59")
    parser.add_argument("--train-log-every", type=int, default=200)
    parser.add_argument("--val-loss-every", type=int, default=4000)
    parser.add_argument("--stall-polls", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.resume:
        if not args.state_path.exists():
            raise FileNotFoundError(f"Missing state file for resume: {args.state_path}")
        runs = load_state(args.state_path)
    else:
        if args.manifest is None:
            raise ValueError("manifest is required unless --resume is set")
        runs = parse_manifest(args.manifest)
        for run in runs:
            submit_run(run, args)
    write_state(args.state_path, runs)
    monitor_runs(runs, args)
    write_state(args.state_path, runs)
    print("[summary] all jobs reached terminal success states")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
