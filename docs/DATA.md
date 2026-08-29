# The data record

The repository holds code and patch coordinates. The results with weight — the
VLM ratings and every map reconstructed from them — are archived separately:

**Zenodo: [10.5281/zenodo.22165217](https://doi.org/10.5281/zenodo.22165217)**

The record contains only what was generated in this work. It does not
redistribute the scene images, the human fixation data or the human meaning
maps: those belong to the studies that collected them.
[DATA_SOURCES.md](DATA_SOURCES.md) says exactly where each of those comes from
and what we did to it, so the inputs can be rebuilt from the original sources.

## Installing it

Unpack the archives over your clone so that `data/` and `results/` sit beside
the notebooks:

```
aimm/
  1_get_patches.ipynb
  src/
  data/            <- saliency maps from the record; scene images and human
                      maps from their original sources (DATA_SOURCES.md)
  results/         <- from the record
```

Every path in the notebooks is relative to the repository root, so no
configuration is needed once the trees are in place. `MANIFEST.csv` in the
record lists every file with its size and SHA-256.

## What is in it

Scope matches the paper: datasets `P21_scegram`, `HH25_indoor` and
`HH25_outdoor`; **no-center-bias (`ncb`) maps only**; the three VLMs reported in
the paper. The `TEST` example is not here — it ships with the repository, images
and outputs included, so the quick start needs no download.

| Path | Contents |
|---|---|
| `results_scores/results_{dataset}_original.parquet` | **The primary result**: every VLM patch rating. One file per dataset; the three models sit side by side in the `model` column, with the prompt text and its hash stored per row. |
| `results_maps/{dataset}/{model}_original/maps_combined_raw/` | Reconstructed AI meaning maps, one per scene |
| `DeepMeaning/{source}__to__{target}/` | DeepMeaning maps, generated here with the code of Hayes & Henderson (2025), for every source/target dataset pair |
| `DeepMeaning_patch_size/` | DeepMeaning at four P21 patch resolutions, for the supplementary figure |
| `data/{dataset}/maps/{gbvs,aws,deepgaze}_ncb/` | Saliency maps, generated here with each model's published code |
| `results_comparisons/` | Correlation outputs written by the notebooks |
| `paper_plots/` | The figures as published |

Maps are `.npy` (raw float32; the meaning maps are on the 1-6 rating scale) plus
a `.png` preview. Read the `.npy`: the `.png` is normalised per image and is not
the same data.

Center-biased (`_cb`) map variants are not included: every analysis in the paper
is `ncb`.

**Completeness.** One combination is slightly short of the full patch count:
HH25-outdoor with Gemma 4 31B IT has 57,926 ratings of 57,936. Those ten patches
are ones where the API returned empty content on all six attempts and the
pipeline moved on - see "When a patch ends up with no score" in
[RUNNING_ON_NEW_DATA.md](RUNNING_ON_NEW_DATA.md), which explains the mechanism
and how to check for it. They are left in place as a worked example of what that
looks like in a real run. Because patches overlap, every pixel of the
reconstructed maps is still covered by other patches at the same scale.

## What you fetch yourself

| Input | From |
|---|---|
| Scene images, all three datasets | The SCEGRAM database; the OSF repository of Hayes & Henderson (2025) |
| Human fixation data and maps | The Zenodo record of Pedziwiatr et al. (2021); the same OSF repository |
| Human meaning maps | The same two sources |

[DATA_SOURCES.md](DATA_SOURCES.md) gives the exact URLs, the paths inside each
archive, the smoothing parameters used to turn the P21 raw data into maps, and
the two scenes whose indoor/outdoor label we changed.

## How much of it you need

You do not need everything to check a result:

- **The ratings alone** (`results_scores/`, ~28 MB) plus the scene images
  rebuild every AI meaning map with section 2 of `2_run_model.ipynb`, without
  spending anything on API calls. That is the cheapest way to verify the
  central claim.
- **The scene images alone** rebuild the patches with `1_get_patches.ipynb`.
- **Figures 2 and 3** additionally need the human meaning maps and fixation
  maps from their original sources, plus the DeepMeaning and saliency maps
  from the record.

## Citing the sources

This work builds on third-party data, and uses third-party code to generate some
of the maps in the record. For the relevant parts, cite the original
publications as well as this paper:

| Component | Cite | Where it is |
|---|---|---|
| P21-scegram fixations and human meaning maps | Pedziwiatr et al. (2021) | Fetch from <https://zenodo.org/record/3490434> |
| P21-scegram stimuli | Öhlschläger & Võ (2017) | Fetch from the SCEGRAM database, <https://www.scenegrammarlab.com/databases/scegram-database/> |
| HH25-indoor / HH25-outdoor stimuli, fixations and human meaning maps | Hayes & Henderson (2025) | Fetch from <https://osf.io/hcnfx/> |
| DeepMeaning maps | Hayes & Henderson (2025) | In the record, generated with the authors' code |
| GBVS saliency maps | Harel et al. (2006) | In the record, generated with the authors' MATLAB implementation |
| AWS saliency maps | Garcia-Diaz et al. (2012) | In the record, generated with the authors' MATLAB implementation |
| DeepGaze IIE maps | Linardos et al. (2021) | In the record, generated with the authors' PyTorch implementation |
| The rating instructions used as our prompt | Henderson & Hayes (2017) | Original instructions at <https://osf.io/654uh/>, under `meaning_mapping/data/rating_instructions` |

## References

Garcia-Diaz, A., Fdez-Vidal, X. R., Pardo, X. M., & Dosil, R. (2012). Saliency
from hierarchical adaptation through decorrelation and variance normalization.
*Image and Vision Computing*, 30(1), 51–64.
<https://doi.org/10.1016/j.imavis.2011.11.007>

Harel, J., Koch, C., & Perona, P. (2006). Graph-Based Visual Saliency. *Advances
in Neural Information Processing Systems*, 19.
<https://papers.nips.cc/paper_files/paper/2006/hash/4db0f8b0fc895da263fd77fc8aecabe4-Abstract.html>

Hayes, T. R., & Henderson, J. M. (2025). DeepMeaning: Estimating and Interpreting
Scene Meaning for Attention Using a Vision-Language Transformer. *Open Mind*, 9,
1020–1036. <https://doi.org/10.1162/opmi.a.6>

Henderson, J. M., & Hayes, T. R. (2017). Meaning-based guidance of attention in
scenes as revealed by meaning maps. *Nature Human Behaviour*, 1(10), 743–747.
<https://doi.org/10.1038/s41562-017-0208-0>

Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). DeepGaze IIE:
Calibrated prediction in and out-of-domain for state-of-the-art saliency
modeling. *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*,
12899–12908. <https://doi.org/10.1109/ICCV48922.2021.01268>

Öhlschläger, S., & Võ, M. L.-H. (2017). SCEGRAM: An image database for semantic
and syntactic inconsistencies in scenes. *Behavior Research Methods*, 49(5),
1780–1791. <https://doi.org/10.3758/s13428-016-0820-3>

Pedziwiatr, M. A., Kümmerer, M., Wallis, T. S. A., Bethge, M., & Teufel, C.
(2021). Meaning maps and saliency models based on deep convolutional neural
networks are insensitive to image meaning when predicting human fixations.
*Cognition*, 206, 104465. <https://doi.org/10.1016/j.cognition.2020.104465>
