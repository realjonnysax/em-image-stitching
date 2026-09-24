# Stitching versions, V1 -> V7

SEM mosaics from a CLEM sample dataset (scans 00001, 00003, 00004, 00005). Tiles are 1280x960 px,
filenames `NNNNN X### Y###.tif` (1-indexed, zero-padded to 3 digits), true overlap
~20%. All mosaics are pyramidal OME-TIFs written by Ashlar. `pixel_size=0.5` is a
placeholder throughout, so Ashlar's `--maximum-shift` (um) converts at 2 px/um.

| Version | Date | Scans | What it was |
|---|---|---|---|
| v1 | 2026-08-20 | all | Original Ashlar wrapper run |
| v2 | 2026-09-24 | 00004 | First re-run; pilot for OME/metadata + seam checks |
| v3 | 2026-09-24 | all | Validated parameter set; 00004 accepted |
| v4 | 2026-09-24 | 00003, 00005 | Registration tweak iteration |
| v5 | 2026-09-24 | 00003, 00005 | User-approved working final |
| v6 | 2026-09-24 | 00003, 00005 | overlap=0.40 attempt - invalid, rejected |
| v7 | 2026-09-28 | all | Patched restitcher (`restitch.py`) |

## V1 - original Ashlar wrapper (2026-08-20)

`<scan>_ashlar.ome.tif`. Notebook loop calling the `ashlar` CLI with
`overlap=0.30`, `--maximum-shift 300` (= 600 px at the placeholder scale),
`--align-channel 0`, and no `--filter-sigma`. Two problems surfaced at the row
seams: duplicated/ghost structures, and a shift budget wide enough (600 px) for
registration to lock onto the wrong feature.

## V2 - pilot re-run (2026-09-24)

`00004_ashlar_v2.ome.tif`. First re-stitch after re-measuring the tile grid;
used to verify OME structure (pyramid levels, PhysicalSizeX, resolution unit)
and to kick off the systematic seam analysis. A `--filter-sigma 8` variant
(fs8) was also tried on 00004 and proved equivalent to 15.

## V3 - validated parameter set (2026-09-24)

`<scan>_ashlar_v3.ome.tif`, all four scans. The parameter set every later
version builds on:

- `overlap=0.20` (nominal step ~1024x768 px on 1280x960 tiles)
- `--filter-sigma 15`
- `--maximum-shift 150` (= 300 px cap at the placeholder pixel size)
- `--align-channel 0`

Correct canvas geometry on all scans; 00004 - which has no true duplicates -
was accepted from v3.

## V4 / V5 - artifact-scan iterations (2026-09-24)

`<scan>_ashlar_v4/v5.ome.tif`, 00003 and 00005 only (the two scans with visible
seam artifacts). Same 0.20-overlap geometry - canvas sizes essentially
unchanged from v3 - with successive registration tweaks while the
duplicate-seam investigation ran. v5 was approved as the working final.

## V6 - overlap=0.40 attempt (2026-09-24) - INVALID

Attempt to fix the seams by declaring more overlap. With a `filepattern` grid
the declared overlap *sets the nominal tile step*, so 0.40 spaced tiles at half
the true separation: canvases shrank (00003 height 9990 -> 7525 px, 00005
11557 -> 10410) and no shift budget could recover the misplacement. Rejected.
Lesson: never "add" overlap by declaring it - fix the registration instead.

## V7 - patched restitcher (2026-09-28)

`<scan>_ashlar_v7.ome.tif`, all four scans. The real fix: patch Ashlar's
*registration* rather than its parameters. `restitch.py` (this repo):

1. `register_padded` replaces `ashlar.utils.register` - whitening + Hann
   window, FFT padded to 2x tile size so large lags cannot wrap around, shift
   search hard-limited to +/-300 px (matching the 150 um cap), 10x cubic
   sub-pixel refinement.
2. `patch_register_pair` replaces `EdgeAligner.register_pair` - multi-scale
   coarse-to-fine pair registration: start at the pair's intersection size,
   double up to 10% of the tile size (cap 50%), keep the scale with the lowest
   error, then score the winning shift with whitened NCC over the actual
   overlap. This guards against false correlation peaks from repetitive EM
   texture and duplicate-structure offsets.
3. CLI wrapper hard-codes the validated v3 parameters (overlap 0.20,
   filter-sigma 15, maximum-shift 150, align-channel 0):
   `python restitch.py <scan> <out.ome.tif> [overlap] [folder]`.

Validation: pyramids/dtype/range intact on all four scans; canvases within ~1%
of v3 (0.20-overlap geometry preserved - unlike v6); aligned NCC vs v3 after
canvas alignment 00001 = 0.971, 00004 = 0.918; 00003/00005 reviewed visually
and confirmed. v7 is the current deliverable for all four scans.

## What stitching still cannot fix

Stage-settling slip at row changes re-images the previous row's bottom strip as
a pixel-accurate copy (plain r = 0.65-0.98), and those duplicated strips are
baked into the raw tiles. No stitcher removes them - affected tiles must be
re-acquired. See the notebook's Y-boundary duplication audit for the confirmed
boundary list.
