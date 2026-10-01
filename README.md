# em-image-stitching

One-click stitching of SEM tile scans on Windows, built on
[Ashlar](https://github.com/labsyspharm/ashlar) with a hardened registration
step for electron-microscopy texture.

Developed at the [Center for Biologic Imaging, University of Pittsburgh](https://www.cbi.pitt.edu)
for a JEOL JSM-IT710HR tile-scan workflow, but it works on any tiled
dataset whose files follow the naming convention below.

## The workflow it replaces

SEM tile scan → manual denoising → invert → export tiles →
manually herd Ashlar. This repo turns all of that into:

**right-click the folder of tiles → Send To → "Stitch with Ashlar"**

Every tile set in the folder is detected and stitched automatically; results
land in `stitched/` as pyramidal OME-TIFs that open directly in ImageJ/Fiji,
napari, or QuPath.

## What's in the repo

| File | What it is |
|---|---|
| `stitch_tool.py` | The tool: auto-detects tile sets in a folder, GUI + command line |
| `stitch_here.bat` | Drop into any folder of tiles, double-click to stitch it |
| `install_sendto.bat` | One-time: adds "Stitch with Ashlar" to the right-click Send To menu |
| `restitch.py` | Standalone Ashlar registration patch (the interesting part, see below) |
| `denoise_tool.py` | SEM denoiser: drops trained U-Net models onto tile folders, GUI + command line |
| `train_denoiser.py` | Trains the denoiser from paired fast/slow scan images (register → normalize → train → eval) |
| `denoise_here.bat` / `install_sendto_denoise.bat` | Same drop-in / Send-To conveniences for denoising |
| `process_here.bat` | All-in-one: denoise a folder (asks whether to invert), then stitch the denoised tiles |
| `install_sendto_process.bat` | One-time: adds "Denoise + Stitch with CBI model" to the right-click Send To menu |
| `STITCHING_VERSIONS.md` | Full V1→V7 development history: what failed, why, and the fix |
| `250325_sem_stitch_notebook...ipynb` | Working notebook: parameter validation, seam and duplicate-structure analysis |

## Tile naming expected

```
00001 X001 Y001.tif
00001 X001 Y002.tif
...
00003 X004 Y013.tif
```

`<set> X<col> Y<row>.tif` — this is how JEOL SEM supporter tile-scan exports
name files (1-indexed, the col/row numbers are zero-padded). Multiple sets in
one folder are fine; each is stitched separately. The tool reports missing
tiles, odd-size tiles, and grid dimensions before stitching.

## Setup

1. Python with Ashlar: `pip install ashlar` (brings numpy/scipy).
   `Pillow` is optional but recommended (enables tile-size sanity checks).
   The GUI uses tkinter, which ships with Python on Windows.
2. Double-click `install_sendto.bat` (stitching), `install_sendto_denoise.bat`
   (denoising), and/or `install_sendto_process.bat` (denoise + stitch) once.

On our lab PCs Ashlar lives in `C:\miniconda3\python.exe`; the `.bat` files
pick that automatically and fall back to whatever `python` is on PATH.

## Usage

Four ways, all equivalent:

```
1. Right-click folder of tiles -> Send To -> "Stitch with Ashlar"
2. Copy stitch_here.bat (+ stitch_tool.py) into the folder, double-click
3. python stitch_tool.py                      (GUI: folder picker, set list, log)
4. python stitch_tool.py <folder> [--scan-only] [--sets 00001,00003]
      [--suffix _v7] [--overwrite] [--overlap 0.20]
```

Output: `stitched/<set>_ashlar_v7.ome.tif` (pyramidal OME-TIFF; suffix
configurable). Use `--scan-only` for a dry-run report of what was found.

## Parameters (validated on real data)

| Parameter | Value | Notes |
|---|---|---|
| `overlap` | 0.20 | **Gotcha:** in `filepattern` mode this *sets the nominal tile step*. It must match the true acquisition overlap — see lessons below |
| `--filter-sigma` | 15 | Whitening filter σ used for registration |
| `--maximum-shift` | 150 | Registration shift cap, in µm |
| `--align-channel` | 0 | Grayscale tiles → channel 0 |
| `pixel_size` | 0.5 | **Placeholder** (hard-coded in `stitch_tool.py`) |

**Pixel-size gotcha:** SEM supporter exports tiles without calibrated physical size, so
`pixel_size=0.5` is a stand-in. Since `--maximum-shift` is interpreted in µm
against it, 150 µm = a **300 px** shift cap. If your pixel size differs,
scale `--maximum-shift` accordingly (or edit `PIXEL_SIZE` in `stitch_tool.py`).

## The registration patch (why stock Ashlar struggles on EM)

Ashlar's default pair registration is an FFT cross-correlation over the tile
overlap region. On repetitive EM texture this fails in two ways, and both
showed up as ghosted/duplicated structures at row seams:

1. **Wrap-around**: unpadded FFTs let a large true shift alias into a small
   (wrong) one.
2. **False peaks**: the narrow overlap region of textured tiles often
   correlates best at the wrong offset.

`restitch.py` patches Ashlar at runtime:

- **Padded FFT** — correlation computed at 2× tile size so large lags cannot
  wrap around, with the search hard-limited to ±300 px (matching the
  maximum-shift cap) and 10× cubic sub-pixel refinement.
- **Multi-scale pair registration** — each tile pair is registered starting
  at its intersection size and doubling the working scale (up to 10% of the
  tile, capped at 50%), keeping the scale with the lowest error; the winning
  shift is then scored by whitened NCC over the actual overlap. Coarse scales
  see the big structure and are far less likely to lock onto repeat texture.

`stitch_tool.py` applies the same patch automatically before stitching, so
output quality matches the validated v7 runs.

## Lessons learned (V1 → V7)

- A shift budget wide enough to reach neighboring features (600 px) lets
  registration lock onto the wrong one — ghosting at seams was caused by the
  budget, not the tiles (v1).
- **Never "fix" seams by declaring more overlap.** With a `filepattern` grid,
  the declared overlap defines the nominal tile step: declaring 0.40 on
  20%-overlap data shrank the canvas by ~25% and no shift budget could
  recover it (v6, rejected).
- The fix is in the *registration*, not the parameters (v7).
- Stage-settling slip at row changes re-images the previous row's bottom
  strip as a pixel-accurate copy baked into the raw tiles. **No stitcher can
  remove these** — affected tiles must be re-acquired. The notebook's
  Y-boundary audit shows how to detect them (near-perfect correlation between
  a tile's bottom strip and its northern neighbor's top strip).

Full history with measurements: [STITCHING_VERSIONS.md](STITCHING_VERSIONS.md).

## Denoising (U-Net)

The repo ships a trained denoiser, `models/sem_denoise_v2.pt` — a compact
U-Net developed at CBI by distillation from high-quality reference-denoised
tiles (65 tile-scan sets, 10,981 image pairs, BED-C detector 6–10 kV,
2.5k–50k magnification). The training pipeline (`train_denoiser.py`) and
training data stay outside the repo on local disks.

The tool auto-picks the model by matching the folder's JEOL sidecar
(instrument, detector, kV, mag, pixel size) against each model's recorded
condition. Any folder of TIFFs works: tiled folders (`<set> X### Y###.tif`)
are processed per tile set, and folders of plain SEM images without tile
coordinates are denoised as a single batch (stitching only applies to tiled
folders).

**Invert option:** SEM images can be shown black-on-white or white-on-black.
Denoised output always follows the raw tile polarity; to flip it, use any of:

- `process_here.bat` / Send To: answer `y` at the "Invert output contrast?" prompt
- GUI: check **Invert**
- CLI: append `--invert`

```
1. Right-click folder of tiles -> Send To -> "Denoise + Stitch with CBI model"
2. Copy process_here.bat (+ the .py tools) into the folder, double-click
3. python denoise_tool.py                     (GUI: folder, model, set list, Invert checkbox)
4. python denoise_tool.py <folder> [--invert] [--sets 00001,00003] [--overwrite]
```

`process_here.bat` chains denoising into stitching: right-click a folder →
denoise → stitch `denoised/` in one run.

## Requirements

- Windows for the `.bat`/Send To conveniences (the Python tool itself is
  cross-platform)
- Python 3.9+ with `ashlar`, `numpy`, `scipy` (`Pillow` optional)
- ~1 min per 100 tiles, single-threaded CPU — no GPU needed

## Credits

Center for Biologic Imaging, University of Pittsburgh.
Built on [Ashlar](https://github.com/labsyspharm/ashlar) (Labsyspharm).
