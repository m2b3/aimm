# Running the pipeline on a new dataset

This walks through producing zero-shot AI meaning maps for scenes of your own.
Nothing in the pipeline is specific to the datasets in the paper: you supply
images and a patch-coordinate file, and the notebooks do the rest.

Throughout, `MYDATA` stands for your dataset name. Use a short name without
spaces — it appears in directory names, file names and the results tables.

---

## 1. What you need

- **Scene images**, all the same pixel size, as `.png` (or `.jpg`).
- **A patch-coordinate CSV** describing where the circular patches go. Section 3
  explains how to build one.
- **An OpenRouter API key** (`OPENROUTER_API_KEY`), and credit on the account.

Optional, and only needed if you want to *compare* your AI maps against
something rather than just produce them:

- human meaning maps, fixation density maps, or saliency maps for the same
  scenes, as `.npy` arrays of the same width and height as the images.

## 2. Directory layout

Create this before you start. Only `images/` and the CSV are yours to provide;
everything else is written by the notebooks.

```
data/MYDATA/
  MYDATA_patches_orig.csv        <- you provide (section 3)
  images/                        <- you provide
    scene001.png
    scene002.png
    ...
  patches_orig/                  <- written by notebook 1
  maps/                          <- optional reference maps, one dir per map type
    fixation/                      scene001.npy, scene002.npy, ...
    meaning_ncb/
```

The file **stem** is the scene identity and ties everything together: a patch
cut from `scene001.png` is named `scene001_w{W}_h{H}_x{X}_y{Y}_r{R}.png`, its map
is written as `scene001.npy`, and a reference map must be `scene001.npy` too.
Keep stems unique, and avoid stems that themselves end in `_w<digits>` — the
reconstruction parses the coordinates back out of the patch filename.

## 3. Specifying the patch coordinates

### The file

`data/MYDATA/MYDATA_patches_orig.csv` holds **one row per patch position**, and
those positions are applied to *every* image in the dataset:

```csv
imw,imh,radius,xc,yc
688,524,53,25,1
688,524,53,83,1
688,524,53,141,1
...
688,524,123,5,20
```

| Column | Meaning |
|---|---|
| `imw`, `imh` | Image size in pixels. Images of a different size are resampled to this before cutting, so it must match your scenes. |
| `radius` | Patch radius in pixels. The saved patch is `2*radius` square with a circular mask. |
| `xc`, `yc` | Patch centre in pixels, origin at the top-left of the image. |

Two patch sizes ("fine" and "coarse") are the convention from the original
meaning-map studies; the file simply concatenates both grids, and the
reconstruction averages within each radius before averaging across radii.
Nothing stops you from using one size or three — every distinct `radius` value
becomes one scale.

Patches near the border are allowed to run off the image. They are padded with
transparency rather than clipped or shifted, which is what Henderson and Hayes
(2017) did and what keeps the map defined all the way to the edge.

### The grid convention

For each radius, the centres form a regular grid that is **symmetric about the
image centre** with an integer spacing. For `n` centres at spacing `step` along
an axis of length `extent`:

```
first   = round((extent - (n - 1) * step) / 2)
centres = first, first + step, first + 2*step, ...
```

### Generating it

`scripts/make_patch_grid.py` writes the file for you:

```sh
python scripts/make_patch_grid.py --name MYDATA --size 1024x768 \
    --fine   r=43  n=20x15 step=51 \
    --coarse r=102 n=12x9  step=85
```

The datasets in the paper are reproduced exactly by:

```sh
# P21-scegram - 688x524, 168 patches per scene
python scripts/make_patch_grid.py --name P21_scegram --size 688x524 \
    --fine r=53 n=12x10 step=58 --coarse r=123 n=8x6 step=97

# HH25 indoor/outdoor - 1024x768, 408 patches per scene
python scripts/make_patch_grid.py --name HH25_indoor --size 1024x768 \
    --fine r=43 n=20x15 step=51 --coarse r=102 n=12x9 step=85
```

### Choosing radius, spacing and counts

The rest of this section applies if you want to follow the reasoning for patch
sizes included in Henderson & Hayes (2017). None of it is enforced by the code:
the pipeline will cut and rate whatever grid you give it. If your question
calls for a different scale, a different sampling density, or a single patch
size, set them deliberately and report them - just be aware that correlations
against maps built on the original convention are then not like for like.

**Radius: match visual angle, not pixels.** A meaning map is a statement about
what an observer could take in around a fixation, so the patch should subtend a
constant visual angle across studies rather than a constant number of pixels.
The original work used roughly **3 degrees for fine and 7 degrees for coarse**
patches. If your scenes span `D` degrees horizontally across `imw` pixels:

```
radius_px = 0.5 * degrees * (imw / D)
```

That is how the P21-scegram radii were derived: 688 px over ~19.7 degrees gives
34.9 px/degree, so 3 degrees gives r ~ 53 px and 7 degrees gives r ~ 123 px. If
your images were never shown to human observers, pick a viewing geometry you
consider representative and state it in your methods.

**Spacing: overlap is what makes the map smooth.** Every pixel should fall
inside several patches at each scale, since the map is the average of the scores
of all patches covering it. In the published datasets the fine grids step by
about 0.55-0.6 patch diameters (roughly 3 patches deep at any pixel) and the
coarse grids by about 0.4 (roughly 6 deep). Staying in that range is the safe
choice: much sparser and the map shows grid artifacts, much denser and you pay
for ratings that add little.

**Counts: cover the image.** `n = round(extent / step)` along each axis is the
starting point; add a row or column if the outermost centres leave the border
thinly covered. Multiply the two axes to get patches per scene, and multiply by
your image count before committing — that product is what you pay for.

### Checking it

Before cutting tens of thousands of patches, look at the grid:

```python
from src.patch_extractor import visualize_patches
visualize_patches("data/MYDATA/MYDATA_patches_orig.csv",
                  "data/MYDATA/images/scene001.png")
```

## 4. Cut the patches - `1_get_patches.ipynb`

Set `DATASET = "MYDATA"` and run. This writes `data/MYDATA/patches_orig/`, one
PNG per (scene x patch position). Expect `n_images * n_rows_in_csv` files: a few
hundred for a pilot, tens of thousands for a full dataset.

## 5. Rate the patches - `2_run_model.ipynb`

Set at the top of the notebook:

```python
DATASET          = "MYDATA"
custom_model     = "google/gemma-4-31b-it"   # any vision model on OpenRouter
custom_prompt_id = "original"                # docs/original_prompt/original.txt
```

**Run a pilot first.** Add `limit=10` to the `async_run_analysis` call, run it,
then check the cost on your OpenRouter activity page and multiply up. One
request is sent per patch, carrying the prompt, the five reference images and
the patch itself; the prompt prefix is marked for provider-side caching, so
after the first few requests the marginal cost is dominated by the patch image.
Rates and image tokenisation differ per model, which is why measuring beats
estimating.

Scores are cached by patch hash + model + prompt id in
`results/results_scores/results_MYDATA_original.parquet`. Re-running after an
interruption only queries what is missing, and adding a second model appends to
the same file under a different `model` value. Delete that parquet only if you
want to pay for everything a second time.

### When a patch ends up with no score

A run can finish with slightly fewer ratings than patches. This is expected and
worth checking for rather than assuming.

The provider call returns nothing usable when the model replies with null
content (an empty completion, or one suppressed by a safety filter), when the
reply cannot be parsed as `{"score": <1-6>}`, or when the request itself errors.
`process_single_image` retries such a patch **six times** with exponential
backoff (1, 2, 4, 8, 16 seconds). If all six attempts come back empty, it logs

```
All 6 attempts failed for <patch>.png - patch will be missing from results
```

and returns `None`. Nothing is written for that patch: the parquet gets no row,
not a null one. So the run reports success, the log line scrolls past, and the
only lasting trace is that the row count is short.

These failures are transient rather than a property of the patch. In the
datasets here, every patch that failed for one model scored normally for the
others, and its neighbours in the same scene and the same minutes went through
fine.

**Check after a run** by comparing the patches on disk against the rows:

```python
import glob
from pathlib import Path
import pandas as pd

on_disk = {Path(p).name for p in glob.glob(f"data/{DATASET}/patches_orig/*.png")}
df = pd.read_parquet(f"results/results_scores/results_{DATASET}_original.parquet")
scored = {Path(str(p).replace("\\", "/")).name
          for p in df.loc[df.model == custom_model, "image_path"]}
print(sorted(on_disk - scored))
```

**The fix is to run the notebook again.** The cache keys on patches already in
the parquet, so a second run re-queries only the missing ones - a handful of
requests, not the whole dataset.

Whether it is worth doing depends on scale. Patches overlap heavily, so a few
missing ratings leave every pixel still covered by other patches at the same
scale; the reconstruction does not break, it just averages over slightly fewer
values in a couple of spots. A run missing entire scenes is a different matter
and should be repeated.

The notebook then reconstructs one map per scene into
`results/results_maps/MYDATA/{model}_{prompt}/maps_combined_raw/` as `.npy`
(raw float32 ratings on the 1-6 scale) and `.png` (a per-image normalised
preview only - see [MAP_FORMATS.md](MAP_FORMATS.md)).

Section 3 of the notebook applies a center-bias template. The analyses in the
paper are all **no-center-bias**, so skip it unless you are deliberately
comparing against models that carry a built-in center bias.

Section 4 correlates the new maps against whatever you put in
`data/MYDATA/maps/`. With no reference maps, drop the other entries from
`select_maps` - the reconstruction in section 2 is already the result.

## 6. Using your own prompt

To change the rating instructions, write a new file
`docs/original_prompt/{your_id}.txt` and set `custom_prompt_id` to its name. Do
not edit `original.txt`: its exact text is hashed into every cached score, and
the pipeline raises on a mismatch rather than silently mixing ratings from two
different prompts.

Reference images are interleaved at the quoted placeholders in the prompt text;
the list of image paths is set in the notebook cell that loads the prompt. The
model is expected to answer with `{"score": <1-6>}` - if you change the response
format, change `_parse_score_response` in `src/meaningfulness.py` to match.

## 7. Comparing against other maps

`src/map_comparison.py` computes per-image Pearson correlations across any set
of map directories:

```python
from src.map_comparison import SaliencyComparator

comparator = SaliencyComparator(
    image_dir="data/MYDATA/images",
    map_configs={
        "aimm":     "results/results_maps/MYDATA/gemma-4-31b-it_original/maps_combined_raw/",
        "mm":       "data/MYDATA/maps/meaning_ncb",
        "fixation": "data/MYDATA/maps/fixation",
    },
)
comparator.load_data()
comparator.compute_correlations(smoothing_sigma=15, exclude_from_smoothing=["fixation"])
```

Only scenes present in *every* configured directory are compared. `sigma=15`
matches Hayes and Henderson (2025); fixation maps are excluded from smoothing
because they are already smoothed density estimates.

If you replace Pearson with a metric that is not invariant to affine rescaling -
KL divergence, information gain, SIM, NSS, AUC - read
[MAP_FORMATS.md](MAP_FORMATS.md) first. The maps arrive at heterogeneous scales,
and that is only harmless for correlation-type metrics.

## 8. Checklist

```
[ ] images all one size, unique stems, in data/MYDATA/images/
[ ] MYDATA_patches_orig.csv written and eyeballed with visualize_patches
[ ] patches per scene x scene count computed, cost sanity-checked
[ ] notebook 1 run, patches_orig/ populated
[ ] notebook 2 run with limit=10, cost checked on OpenRouter
[ ] notebook 2 run in full, parquet + maps_combined_raw/ written
[ ] maps inspected as .npy, not the .png previews
```
