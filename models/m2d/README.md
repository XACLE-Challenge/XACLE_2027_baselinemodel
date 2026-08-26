# M2D-CLAP runtime

This directory uses the official `PortableM2D` runtime from
<https://github.com/nttcslab/m2d> at commit
`3d0c4de9447c404a8d3f9f37e04f53bc902e09b3`.
The vendored file has a small integration patch that passes the configured
Hugging Face cache directory to its text encoders.

Download the official M2D-CLAP 2025 standard checkpoint and place it at:

```text
models/m2d/checkpoints/
  m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/
    checkpoint-30.pth
```

Checkpoint files are intentionally excluded from Git.

The downloaded `checkpoint-30.pth` used for this baseline has SHA-256:

```text
238521603c04862ab151cdd80980b591cb36ebe844d43203992fac9ef085c8a1
```
