# Publication checklist

## Completed

- [x] Separate the XACLE 2027 model from the XACLE 2026 reference code.
- [x] Exclude datasets, virtual environments, caches, run outputs, and M2D
      checkpoints from Git.
- [x] Include M2D portable runtime provenance and its upstream license.
- [x] Save only the trainable prediction head in XACLE checkpoints.
- [x] Add config validation, fixed random seeds, unit tests, and an integration
      smoke test.
- [x] Verify the macOS CPU train/inference/evaluation path.
- [x] Confirm that no Git-tracked file exceeds GitHub's 100 MB file limit.
- [x] Remove the XACLE 2026 BYOL-A/RoBERTa implementation and pretrained
      checkpoint from the public source tree. They remain recoverable from the
      initial Git commit for historical reference.

## Required before release

- [ ] Run a full training job on ABCI/CUDA and record the environment and
      results.
- [ ] Pin the final CUDA/PyTorch dependency set used on ABCI.
- [ ] Add the official XACLE 2027 dataset and evaluation instructions when
      they are finalized.
- [ ] Add the final baseline metrics and citation information.
- [ ] Review repository history for credentials, private URLs, and unintended
      large artifacts before pushing the release branch.
