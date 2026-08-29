"""
Helpers for the pipeline-schematic figure (Figure 1 in ``3_paper_plots.ipynb``).

Pure, notebook-independent functions: reading patch coordinates, cropping a
circular patch, reconstructing a per-radius meaningfulness map for one scene,
and looking up a single patch's VLM score.  The figure assembly/layout stays in
the notebook; only the reusable logic lives here.

Dependencies: numpy, pandas, Pillow
"""

import csv
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def read_patch_params(csv_path):
    """Read patch coordinates from a ``{dataset}_patches_orig.csv`` file.

    Returns a list of dicts with integer ``imw, imh, radius, xc, yc`` fields.
    """
    params = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            params.append({k: int(row[k]) for k in ("imw", "imh", "radius", "xc", "yc")})
    return params


def circular_patch(image, x, y, r, scale=1.0):
    """Square RGBA patch with a circular crop (white outside)."""
    img = image.convert("RGBA")
    w, h = img.size
    sl, st, sr, sb = x - r, y - r, x + r, y + r
    cl, ct, cr, cb = max(0, sl), max(0, st), min(w, sr), min(h, sb)
    size = r * 2
    patch = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    patch.paste(img.crop((cl, ct, cr, cb)), (cl - sl, ct - st))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    result = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    result.paste(patch, (0, 0), mask)
    if scale != 1.0:
        result = result.resize((int(size * scale), int(size * scale)), Image.LANCZOS)
    return result


def reconstruct_scene_map(df, scene, model, prompt_id, radius):
    """Per-radius meaningfulness map for one scene (mirrors patch_reconstruction25)."""
    scene_df = df[
        (df["model"] == model)
        & (df["prompt_id"] == prompt_id)
        & (df["image_path"].str.contains(scene, na=False))
    ]

    def parse_path(p):
        stem = re.sub(r"\.\w+$", "", re.sub(r".*[\\/]", "", p))  # basename, no ext
        m = re.match(r"^(.+?)_w(\d+)_h(\d+)_x(\d+)_y(\d+)_r(\d+)$", stem)
        return m.groups() if m else None

    rows = []
    for _, row in scene_df.iterrows():
        parsed = parse_path(row["image_path"])
        if parsed:
            rows.append(dict(w=int(parsed[1]), h=int(parsed[2]), x=int(parsed[3]),
                             y=int(parsed[4]), r=int(parsed[5]), score=float(row["score"])))
    pf   = pd.DataFrame(rows)
    r_df = pf[pf["r"] == radius]
    orig_w, orig_h = int(r_df["w"].iloc[0]), int(r_df["h"].iloc[0])

    accumulated = np.zeros((orig_h, orig_w))
    count       = np.zeros((orig_h, orig_w))
    gy, gx    = np.ogrid[-radius:radius, -radius:radius]
    circ_mask = gx**2 + gy**2 <= radius**2
    for _, p in r_df.iterrows():
        cx, cy, val = int(p["x"]), int(p["y"]), float(p["score"])
        sl, st, sr, sb = cx - radius, cy - radius, cx + radius, cy + radius
        cl, ct, cr, cb = max(0, sl), max(0, st), min(orig_w, sr), min(orig_h, sb)
        if cl >= cr or ct >= cb:
            continue
        ph, pw = cb - ct, cr - cl
        pm = circ_mask[ct - st:ct - st + ph, cl - sl:cl - sl + pw]
        if pm.shape != (ph, pw):
            pm = np.ones((ph, pw), dtype=bool)
        accumulated[ct:cb, cl:cr][pm] += val
        count[ct:cb, cl:cr][pm]       += 1
    return np.where(count > 0, accumulated / count, np.nan).astype(np.float32)


def patch_score(df, scene, x, y, r):
    """Actual VLM score for one patch (scene-filtered to avoid cross-scene hits)."""
    scene_df = df[df["image_path"].str.contains(scene, na=False)]
    pat = re.compile(rf"_x{x}_y{y}_r{r}(?:\D|$)")
    m = scene_df[scene_df["image_path"].apply(lambda p: bool(pat.search(p.replace("\\", "/"))))]
    return int(m["score"].iloc[0]) if len(m) else "?"
