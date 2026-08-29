# Where the input data comes from

The stimuli, human fixation data and human meaning maps used in this work belong
to the studies that collected them, and are **not** redistributed here or in our
data record. This page says exactly where each one comes from and what we did to
it, so the inputs can be rebuilt from the original sources.

What our record does contain — VLM ratings, reconstructed AI meaning maps,
DeepMeaning maps, saliency maps, comparison outputs and figures — is described in
[DATA.md](DATA.md).

---

## P21-scegram

| Input | Where to get it |
|---|---|
| Scene images | The SCEGRAM database: <https://www.scenegrammarlab.com/databases/scegram-database/> (Öhlschläger & Võ, 2017). We use the 36 scenes of the "consistent" (CON) condition. |
| Fixation data | Pedziwiatr et al. (2021), <https://zenodo.org/record/3490434> — `eye_movements_scegramSubset_preProcessed_fin_18-Jun-2019 12_01_41/coreImStr.mat` |
| Human meaning ratings | Pedziwiatr et al. (2021), <https://zenodo.org/record/3490434> — `4a1_scegramSubset_processedRatings_SUR_FIN_18-Jun-2019 14_18_33/subsetScegram_ratingsTabAgg_18-Jun-2019 14_18_33.csv` |

**Fixation maps.** Built here from the raw fixation data, for all 36 images, by
Gaussian smoothing with a **6 dB cutoff frequency** (`GAUSSIAN_CUTOFF = 6`), as
in Pedziwiatr et al. (2021). The result goes in `data/P21_scegram/maps/fixation/`.

**Human meaning maps.** Built here from the raw ratings, for all 36 images, with
`mfiles/P21_stitch_meaning_maps.m`. The result goes in
`data/P21_scegram/maps/meaning_ncb/`.

## HH25-indoor and HH25-outdoor

All three inputs come from the DeepMeaning repository of Hayes & Henderson
(2025), <https://osf.io/hcnfx/>:

| Input | Path inside that repository |
|---|---|
| Scene images | `DeepMeaning/data/scenes/internal/` |
| Fixation maps | `DeepMeaning/data/attention/` |
| Human meaning maps | `DeepMeaning/data/human_meaning/average_rating_maps/` |

**Fixation maps** were already smoothed at the source; the smoothing parameter is
not specified there. They cover a subset of scenes (50 indoor, 50 outdoor as
distributed), not the full set. Copy them in unchanged.

**Human meaning maps** are distributed for all 282 scenes together, unsmoothed.
Smoothing is applied later in our pipeline (sigma = 15, only for the
fixation-prediction analyses), not to the stored maps. Copy them in unchanged.

### The indoor/outdoor split differs from the source

Two scenes, **`target_greenhouse`** and **`target_porch`**, are labelled
inconsistently between analyses in Hayes & Henderson (2025). We inspected them
and assigned **both to indoor**. Every count in our datasets follows from that:

| | As distributed | Here |
|---|---|---|
| Indoor scenes | 139 | **140** (+ `target_greenhouse`) |
| Outdoor scenes | 143 | **142** |
| Indoor scenes with fixation maps | 50 | **51** (+ `target_porch`) |
| Outdoor scenes with fixation maps | 50 | **49** |

So when rebuilding `data/HH25_indoor/` and `data/HH25_outdoor/`, move those two
scenes — and `target_porch`'s fixation map — into the indoor set.

## Saliency maps

`gbvs_ncb/`, `aws_ncb/` and `deepgaze_ncb/` are **not** third-party data: we
generated them from the scene images with the model authors' published code, and
they are included in our data record.

They can also be regenerated once the images are in place. The models themselves
are not distributed here — only thin wrappers that call them, so each one has to
be obtained from its authors first:

| Maps | Model | Wrapper here |
|---|---|---|
| GBVS | <https://github.com/Pinoshino/gbvs> (Harel et al., 2006) | `mfiles/get_gbvs_maps.m`, which sets `unCenterBias = 1` |
| AWS | <http://persoal.citius.usc.es/xose.vidal/research/aws/AWSmodel.html> (Garcia-Diaz et al., 2012) | `mfiles/get_aws_maps.m` |
| DeepGaze IIE | <https://github.com/matthias-k/DeepGaze> (Linardos et al., 2021) | `scripts/get_deepgaze_maps.py`, which uses a uniform all-zero bias template |

The `ncb` settings in that table matter: GBVS and DeepGaze IIE both carry an
image-independent centre bias by default, and the paper switches it off in both
so that they can be compared fairly against the meaning maps.

GBVS and AWS are MATLAB, and you supply them yourself. DeepGaze IIE is the
exception: its dependencies (`deepgaze-pytorch` and OpenAI's `CLIP`, both from
git) are already declared in `pyproject.toml`, so `uv sync` installs them for
you. They are needed *only* by `scripts/get_deepgaze_maps.py` — the notebooks
import neither, nor torch — and the DeepGaze maps ship in the data record, so
reproducing the paper never runs this script.

## After assembling the inputs

The directory layout the notebooks expect is in
[RUNNING_ON_NEW_DATA.md](RUNNING_ON_NEW_DATA.md) §2. With `images/` in place,
`1_get_patches.ipynb` cuts the patches, and the VLM ratings in our record can be
turned straight into meaning maps by section 2 of `2_run_model.ipynb` — no API
calls needed to reproduce the published maps.

## References

See the References section of [DATA.md](DATA.md) for the full citations, and
cite the original publications for any of these inputs that you use.
