# Windows viewer package

Status: **current**. Audience: you (packaging) and friends receiving the zip.

## Build the zip (your machine)

From WSL (downloads llvm-mingw once into `~/.cache`, then cross-compiles):

```sh
tools/package_viewer_windows.sh
```

Output: **`dist/NeuralVFX-Viewer-1.0.0-win64.zip`** (`NVFX_VIEWER_VERSION` overrides the version) with **`nvfx_viewer.exe`**, `library/`, `models/3d/`, `data/` (all models from `models/restore.sh`), sample scenes, and **`Run NeuralVFX Viewer.bat`**.

## For your friend

- **64-bit Windows 10/11**, current GPU drivers (OpenGL 3)
- Unzip, run **`nvfx_viewer.exe`** or **`Run NeuralVFX Viewer.bat`**
- Effects on **CPU (AVX2)**; AMD/NVIDIA/Intel GPU only draws the window
- Keep `library\`, `models\`, `data\`, and `examples\` next to the exe
