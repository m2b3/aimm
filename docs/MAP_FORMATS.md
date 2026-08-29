# Map formats, scaling, and which metrics are safe

Maps enter the comparison at **heterogeneous scales**. This is safe for the
analyses in the paper because they use per-image Pearson correlation, which is
invariant to positive affine rescaling. It is not safe in general, and this page
exists so that anyone swapping in a different metric knows what to check first.

## Two files, one array

`patch_reconstruction25.reconstruct_images` writes each map twice:

| File | What it is |
|---|---|
| `.npy` | The map. `float32`, raw values on the 1-6 rating scale. |
| `.png` | An 8-bit **preview**, min-max normalised *per image*. Not the same data. |

`SaliencyMapLoader` resolves each scene stem in the order `.npy` > `.mat` >
`.png`, so the raw values are used wherever they exist. Read the `.npy`. The
`.mat` step is legacy support for older map directories; nothing in the
pipeline writes `.mat` any more.

## What each map type is on disk

| Map | On disk (ncb) | Loaded from | Values |
|---|---|---|---|
| `fixation` | npy + png | `.npy` | Raw density. P21 ~1e-2, HH25 ~1e2, including tiny negative values from the kernel density estimate. |
| `mm` (human meaning) | npy + png | `.npy` | Raw, 1-6 rating scale |
| `aimm` (this pipeline) | npy + png | `.npy` | Raw, 1-6 rating scale |
| `dmm` (DeepMeaning) | npy + png | `.npy` | Raw, 1-6 scale, visibly compressed range (never reaches 1.0 or 6.0) |
| `gbvs`, `aws`, `deepgaze` | png only | `.png` | Min-max normalised to exactly `[0, 1]` in every file, 8-bit quantised |

The three saliency models exist only as PNGs, so their absolute level carries no
information and every file spans the full 0-1 range by construction.

## Why Pearson is fine here

`SaliencyComparator` computes one correlation matrix per image and aggregates
the resulting `r` values across images. Within an image, both the cross-map
scale differences and the per-image min-max normalisation are positive affine
transforms, which Pearson cancels exactly (measured max |delta r| ~ 3e-16). The
8-bit quantisation of the PNG-sourced maps is the only non-affine difference and
moves `r` by ~3e-4. The optional smoothing (Gaussian, then rescale to the
original range) and histogram matching (CDF-based) are scale-free as well.

## Before switching metrics

Pearson and Spearman are the exception, not the rule.

- **KL divergence, information gain, SIM / histogram intersection** need true
  probability densities that sum to 1. Re-normalise from the raw `.npy`, and
  clip the fixation maps' small negative values first. PNG-sourced maps cannot
  be used as densities at all.
- **NSS and AUC** need raw fixation *coordinates*, not the density maps in
  `maps/fixation/`.
- **Any pooled or global analysis** — concatenating pixels across images into
  one correlation — and any **cross-image comparison of absolute
  meaningfulness** is broken by per-image min-max normalisation. Verified on
  P21-scegram: pooled `r` = 0.637 from raw values versus 0.651 from
  per-image-normalised ones. The cross-image mean AI meaning level spans
  2.90-4.92 on the 1-6 scale: real signal, present in the `.npy`, erased in the
  `.png`.

**Rule of thumb:** always read the `.npy` where it exists, and treat the `.png`
as a preview. If a metric needs the saliency-model maps at a raw scale, they
have to be regenerated from the original model code rather than recovered from
the PNGs.
