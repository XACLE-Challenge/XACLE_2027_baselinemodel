# ABCI 3.0 execution

These scripts target one shared H-series GPU (`rt_HG`) on ABCI 3.0.

The verified environment uses Python 3.12.9, PyTorch 2.13.0 with CUDA 13.0,
and an NVIDIA H200 GPU. WAV files are loaded with SoundFile so execution does
not depend on TorchCodec or system FFmpeg libraries.

## Files that must exist on ABCI

- This repository
- `models/m2d/checkpoints/.../checkpoint-30.pth`
- `hf_cache/` containing the cached M2D-CLAP text encoder files
- `datasets/XACLE_dataset`, which may be a symbolic link to an existing copy

## Setup

From the project root on an ABCI login node:

```bash
ln -s /home/acg17087nj/xacle/XACLE_benchmark/datasets/XACLE_dataset \
  datasets/XACLE_dataset
bash scripts/abci/setup_env.sh
```

## Jobs

Run the short GPU check first:

```bash
qsub scripts/abci/smoke.pbs
```

After it succeeds, submit full training:

```bash
qsub scripts/abci/train.pbs
```

Evaluate the verified best run on validation and test data:

```bash
qsub scripts/abci/evaluate_best.pbs
```

`evaluate_best.pbs` uses the low-point `gch51642` group, requests one shared
GPU for at most 10 minutes, and writes metrics under the run's `evaluation/`
directory. Update `BEST_RUN_DIR` in the script when evaluating another run.

## C/D split experiment

Place `train_C.csv`, `validation_C.csv`, `test_C.csv`, `train_D.csv`,
`validation_D.csv`, and `test_D.csv` under:

```text
datasets/XACLE_dataset/meta_data/experiments/
```

Validate the metadata before requesting a GPU:

```bash
python scripts/experiments/validate_cd_metadata.py
```

Then submit both training/evaluation pipelines in one allocation:

```bash
qsub scripts/abci/run_cd_experiment.pbs
```

The job trains Model C and Model D independently, selects each best checkpoint
using validation SRCC, evaluates the matching validation/test splits, and
writes the combined results to `chkpt_cd/summary.csv`. It uses `gch51642` and
requests one shared GPU for at most 15 minutes.

Use `qstat` to check the job state. PBS writes the merged stdout/stderr log in
the directory from which the job was submitted.
