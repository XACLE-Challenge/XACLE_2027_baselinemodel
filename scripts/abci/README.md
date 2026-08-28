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

Use `qstat` to check the job state. PBS writes the merged stdout/stderr log in
the directory from which the job was submitted.
