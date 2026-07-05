# Distillation Artifacts

Downloaded KataGo engines, model weights, and release archives live under this
folder, but the artifact subdirectories are intentionally ignored by git:

- `downloads/`: cached release archives.
- `engine/`: extracted KataGo engine packages.
- `models/`: downloaded KataGo neural net weights.

The phase-1 generator auto-detects a host-appropriate executable under
`Distillation/engine/*/`: `katago.exe` on Windows and `katago` on Linux/macOS.
You can still pass `--katago` explicitly when choosing between multiple local
engines.

For Linux portability, use the official KataGo v1.16.5 OpenCL x64 release:

```powershell
Invoke-WebRequest `
  -Uri https://github.com/lightvector/KataGo/releases/download/v1.16.5/katago-v1.16.5-opencl-linux-x64.zip `
  -OutFile Distillation/downloads/katago-v1.16.5-opencl-linux-x64.zip
Expand-Archive `
  -Path Distillation/downloads/katago-v1.16.5-opencl-linux-x64.zip `
  -DestinationPath Distillation/engine/katago-v1.16.5-opencl-linux-x64
```

For a Linux machine with matching NVIDIA CUDA/TensorRT libraries, the faster
TensorRT asset can also be installed and selected with `--katago`.
