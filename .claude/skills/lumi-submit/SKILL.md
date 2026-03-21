# LUMI Submit Skill

## What This Does
Submit and manage Parameter Golf experiments on LUMI supercomputer.

## How to Use

### Submit a training job
```bash
scripts/lumi.sh submit scripts/slurm/lumi_train.sh
```

### With custom env vars (passed through to train_gpt.py)
```bash
ssh lumi "cd ~/parameter-golf && NUM_GPUS=8 RUN_ID=my_run sbatch scripts/slurm/lumi_train.sh"
```
Or use lumi.sh submit with the script.

### Check job status
```bash
scripts/lumi.sh status
```

### Read logs
```bash
scripts/lumi.sh logs <job_id>
scripts/lumi.sh errors <job_id>
scripts/lumi.sh tail <job_id>
```

### Check partition availability (if jobs are pending)
```bash
scripts/lumi.sh info
```

### For crashed/failed jobs
```bash
scripts/lumi.sh sacct <job_id>
scripts/lumi.sh errors <job_id>
```

## Gotchas
- standard-g is EXCLUSIVE: every job gets full 8-GCD node regardless of --gpus-per-node. Use lumi_packed.sh for multiple 1-GPU experiments.
- If standard-g is full, try dev-g (3h max but often has idle nodes).
- Container path: /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
- Dataset must be on scratch first. Run lumi_setup_data.sh once before first training.
- Logs are in ~/parameter-golf/logs/ on LUMI.
- Always sync code before submitting: scripts/lumi.sh sync
