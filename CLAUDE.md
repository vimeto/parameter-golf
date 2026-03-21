# Parameter Golf - Claude Code Guidelines

## Commits

ALWAYS use 1-line commit messages. NO co-authorship lines. Example:
```
git commit -m "feat: add slurm training scripts for LUMI"
```

## Project

OpenAI Parameter Golf challenge: train the best language model (lowest bits-per-byte) that fits in a 16MB artifact and trains in 10 min on 8xH100 SXM.

- Repo: https://github.com/vimeto/parameter-golf (fork of openai/parameter-golf)
- Upstream: https://github.com/openai/parameter-golf
- Research: see `RESEARCH_ANALYSIS.md` in parent dir (`~/code/omat/parameter-golf/`)
- Plan: see `TODO.md` in parent dir

## Key Constraints

- Artifact (code + compressed model) <= 16,000,000 bytes
- Training <= 10 min on 8xH100 SXM
- Evaluation <= 10 min on 8xH100 SXM (separate budget, can train on val data)
- BPB metric on FineWeb validation set

## CRITICAL: Code Modification Rules

- **NEVER modify `records/our_submission/` for experimentation.** Multiple code paths must run concurrently.
- The train script on LUMI (`~/parameter-golf/train_gpt.py`) is the single source that ALL runs execute. Feature flags (env vars) control which code paths are active.
- To add a new feature: edit `train_gpt.py` (root), gate it behind an env var, sync to LUMI, then reference the env var in spec files.
- Exploration runs are 8 parallel 1-GPU jobs — each with different env vars — sharing the SAME code.

## Development Workflow

1. Edit `train_gpt.py` (root) — gate new features behind env vars
2. Local smoke test via MLX (200 steps, ~15 min on M2 Max)
3. `scripts/lumi.sh sync` to push code to LUMI
4. Create spec file in `specs/` with 8 experiment configs
5. Submit to LUMI via `scripts/lumi.sh submit ...`
6. Read results via `scripts/lumi.sh logs <job_id>`
7. Record results in `experiments/results.tsv`
8. Follow research directions in parent `TODO.md`

## LUMI Supercomputer

- GPUs: AMD MI250X (8 GCDs per node, 64GB HBM2e each)
- Partition: `standard-g` (up to 2 days), `dev-g` for quick tests (3h max)
- Account: `project_462001163`
- Code dir on LUMI: `~/parameter-golf`
- Data on scratch: `/scratch/project_462001163/vtoivone/pgolf_cache/`
- Container: `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`

## LUMI Training Guidelines

- **ALWAYS use torch.compile** — enabled via `shape_padding=False` + `fullgraph=False` in our ROCm fork. Gives ~2x speedup over eager mode.
- **Token-parity with 8xH100 requires ~41 min on 8xMI250** — the benchmark processes 6.8B tokens in 10 min on 8xH100 (11.4M tok/s). Our 8xMI250 with compile does 2.80M tok/s (187ms/step with 524K batch), so set `MAX_WALLCLOCK_SECONDS=2500` (~42 min) for comparable runs. We are 4.1x slower than 8xH100.
- For quick exploration sweeps, use `lumi_packed.sh` with 8x 1-GPU runs (50 min each, ~360M tokens per run — directional signal only).
- For proper validation runs, use `lumi_sequential.sh` with 8 GPUs and 75 min wallclock per experiment.
- `dev-g` has faster queue times than `standard-g` — use for quick tests.

## Autoresearch Loop

Two-tier continuous experimentation:
- **Node A (exploration):** 8x 1-GPU packed runs, 50min train + 10min eval. Directional signal only.
- **Node B (validation):** 1x 8-GPU full node, 45min train (2700s). Token-parity with H100 benchmark.

Both nodes run in parallel. After each exploration batch, the best config is promoted to validation.

State tracked in:
- `experiments/state.json` — current phase, best scores, running jobs
- `experiments/exploration_log.md` — all exploration batches with results and promotion decisions
- `experiments/validation_log.md` — all validation runs with comparison to global best

Every run with a feature flag must pass an integrity check (specific log line proving the feature is active).
See `.claude/skills/experiment/SKILL.md` for the full protocol.

## Local Testing (M2 Max, 64GB)

```bash
cd ~/code/omat/parameter-golf
source .venv/bin/activate
RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=32768 VAL_LOSS_EVERY=0 \
  VAL_BATCH_SIZE=8192 TRAIN_LOG_EVERY=20 MAX_WALLCLOCK_SECONDS=0 \
  python3 train_gpt_mlx.py
```

Local numbers are ~2-3x worse than H100 due to undertraining. Use for directional comparison only.

## File Structure

- `train_gpt.py` — Baseline training script (CUDA)
- `train_gpt_mlx.py` — Baseline training script (MLX/Mac)
- `records/our_submission/` — Our working submission
- `scripts/lumi.sh` — LUMI SSH bridge
- `scripts/slurm/` — SLURM job scripts
- `experiments/results.tsv` — Experiment tracking
- `experiments/runs/` — Saved improved run artifacts

## Skills

- **lumi-submit** (`.claude/skills/lumi-submit/SKILL.md`) — Submit and manage jobs on LUMI supercomputer
- **experiment** (`.claude/skills/experiment/SKILL.md`) — Autoresearch experiment loop: propose, test, measure, record
- **local-test** (`.claude/skills/local-test/SKILL.md`) — Quick MLX smoke tests on M2 Max
