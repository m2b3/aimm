# AIMM — zero-shot AI meaning maps

Code and analysis for:

> **Human-like meaning maps from single-prompt VLM ratings of local scene meaning**
>
> Katarzyna Jurewicz, Yohaï-Eliel Berreby, B. Suresh Krishna
>

Meaning maps describe where the semantically informative regions of a scene are.
They are normally built by showing small circular scene patches to
human raters. This repository allows you to obtain the ratings with a vision-language
model that receives the *same* written instructions used with human participants,
in a single prompt, with no fine-tuning and no task-specific training data
— and reproduces every figure in the paper from those ratings.

The pipeline is dataset-agnostic. To run it on scenes of your own, see
**[docs/RUNNING_ON_NEW_DATA.md](docs/RUNNING_ON_NEW_DATA.md)**. To reproduce the
paper, you need the scene images and human data from their original sources —
**[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** says exactly where each one is.
Every other map the analyses use — the AI meaning maps, DeepMeaning and the
saliency models — you can either generate from scratch or download ready-made
from the Zenodo record (see [Data](#data)).

## Installation

```sh
git clone https://github.com/m2b3/aimm.git
cd aimm
uv sync
```

Querying a model needs an [OpenRouter](https://openrouter.ai) API key:

```sh
export OPENROUTER_API_KEY=sk-or-...        # Windows: setx OPENROUTER_API_KEY sk-or-...
```

## Quick start

The repository ships the patch coordinates and the small `TEST` example
dataset; the paper's images and maps it does not (see [Data](#data)). `TEST`
exercises the whole pipeline for a few cents of API credit (you can also use
the pre-computed example ratings in `results_TEST_original.parquet`):

```sh
uv run jupyter lab
```

1. `1_get_patches.ipynb` — set `DATASET = "TEST"`, run. Cuts the circular patches
   out of `data/TEST/images/` using `data/TEST/TEST_patches_orig.csv`.
2. `2_run_model.ipynb` — set `DATASET = "TEST"`, run. Rates every patch 1–6 with
   the chosen VLM, reconstructs one meaning map per scene, and correlates it
   against the reference map — DeepGaze IIE for `TEST`, which has no human
   meaning maps; the human maps for the paper's datasets.

That is the entire method. Notebooks 3 and 4 are the paper's figures.

## Pipeline

| Notebook | Does | Writes |
|---|---|---|
| `1_get_patches.ipynb` | Cuts circular patches from each scene at the coordinates in `{DATASET}_patches_orig.csv` | `data/{DATASET}/patches_orig/` |
| `2_run_model.ipynb` | Rates every patch with a VLM, reconstructs per-scene maps, optional center bias, quick correlation check | `results/results_scores/`, `results/results_maps/` |
| `3_paper_plots.ipynb` | Figures 1–3 and their statistics tables | `results/paper_plots/` |
| `4_supplementary_plots.ipynb` | Supplementary figure: DeepMeaning patch-size sweep, five source → target pairs | `results/paper_plots/` |
| `appendix_plots.ipynb` | Appendix figure: the prompt's reference images | `results/paper_plots/` |

Ratings are cached by patch hash + prompt id, so re-running a notebook only
queries patches that have not been scored yet. **API calls cost money** — use
`limit=` and confirm the model, prompt and dataset before a full run.

### Figures

| Figure | Notebook | File |
|---|---|---|
| Fig. 1 — pipeline schematic | `3_paper_plots.ipynb` | `fig1_schematic.png` |
| Fig. 2 — AIMM vs human meaning maps, three VLMs and DeepMeaning | `3_paper_plots.ipynb` | `fig2_model_comparison_violin.png` |
| Fig. 3 — fixation prediction vs HMM, DMM, GBVS, AWS, DGIIE | `3_paper_plots.ipynb` | `fig3_fixation_correlations_{model}.png` |
| Fig. S1 — DeepMeaning patch-size sweep, five source → target pairs | `4_supplementary_plots.ipynb` | `supplementary/figS1_resolution_comparison_violin.png` |
| Appendix — prompt reference images | `appendix_plots.ipynb` | `appendix_prompt_examples.png` |

## Data

The VLM ratings, the reconstructed AI meaning maps, the DeepMeaning maps and the
saliency maps are archived separately:

**Zenodo: [10.5281/zenodo.22165217](https://doi.org/10.5281/zenodo.22165217)**

Unpack it over your clone so that `data/` and `results/` sit next to the
notebooks. [docs/DATA.md](docs/DATA.md) lists what the record contains.

The record holds only what was generated in this work. The scene images, human
fixation data and human meaning maps belong to the studies that collected them
and are not redistributed here; fetch them from the sources below.
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) gives the exact URLs, the paths
inside each archive, and what we did to the raw data to produce the maps the
notebooks read.

| Dataset | Images and human data from | Scenes | Scenes with fixations | Patches/scene |
|---|---|---|---|---|
| `P21_scegram` | Pedziwiatr et al. 2021 ([Zenodo](https://zenodo.org/record/3490434)); stimuli from the [SCEGRAM database](https://www.scenegrammarlab.com/databases/scegram-database/) | 36 | 36 | 168 |
| `HH25_indoor` | Hayes & Henderson 2025 ([OSF](https://osf.io/hcnfx/)) | 140 | 51 | 408 |
| `HH25_outdoor` | Hayes & Henderson 2025 ([OSF](https://osf.io/hcnfx/)) | 142 | 49 | 408 |
| `TEST` | Ships with this repository ([sources](data/TEST/SOURCES.md)) | 2 | — | 168 |

## The prompt

`docs/original_prompt/original.txt` is the single prompt behind every result in
the paper. It is the instruction text given to human raters by Henderson & Hayes
(2017), with the five reference images in `docs/original_prompt/` interleaved at
the points where the original instructions showed them. The model returns
`{"score": <1-6>}`.

The original instructions are at <https://osf.io/654uh/>, under
`meaning_mapping/data/rating_instructions`.

The prompt text is hashed into every cached rating. Editing a prompt file in
place invalidates that cache and the pipeline will refuse to run against it —
put a reworded prompt in a new `docs/original_prompt/{id}.txt` and point
`custom_prompt_id` at it instead.

## Models

The paper reports three models, all queried through OpenRouter at
`temperature=0.0`:

| Model | OpenRouter slug | Weights |
|---|---|---|
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | proprietary |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` | proprietary |
| Gemma 4 31B IT | `google/gemma-4-31b-it` | open |

Figure 3 uses Gemma 4 31B IT: its weights are public, so those results stay
reproducible after the proprietary endpoints change. Any other vision model on
OpenRouter works — set `custom_model` in `2_run_model.ipynb`.

## Modules

| Module | Role |
|---|---|
| `src/meaningfulness.py` | Async VLM querying: caching, rate limiting, hash-based dedup |
| `src/patch_extractor.py` | Cuts circular/square patches per the coordinate CSV |
| `src/patch_reconstruction25.py` | Rebuilds per-scene maps from patch scores |
| `src/map_comparison.py` | Per-image Pearson/Spearman correlation between maps |
| `src/paper_plots.py` | Multi-dataset publication figures |
| `src/stats_utils.py`, `src/schematic_fig.py` | Statistics helpers; Figure 1 panels |

Maps are written as `.npy` and `.png` (**preview**).
See [docs/MAP_FORMATS.md](docs/MAP_FORMATS.md) — data format matters
if you swap Pearson correlation for a metric that is not scale-free.

## Citation

To cite this work, see `CITATION.cff`.

The record does not redistribute third-party data, but the analyses depend on
it, and some maps in the record were generated with third-party code — both
should be cited alongside this paper. [docs/DATA.md](docs/DATA.md) has the full
overview: which publication to cite for which part, and the complete
references.

## License

Code is MIT (see `LICENSE`). The data record is licensed separately. The scene
images, fixation data and human meaning maps stay with their original publications,
under their own licenses; see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
