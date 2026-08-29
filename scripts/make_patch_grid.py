#!/usr/bin/env python
"""Generate a patch-coordinate CSV for a new dataset.

    python scripts/make_patch_grid.py --name MYDATA --size 1024x768 \
        --fine   r=43  n=20x15 step=51 \
        --coarse r=102 n=12x9  step=85

Writes ``data/MYDATA/MYDATA_patches_orig.csv``, the file 1_get_patches.ipynb
reads to cut patches out of every image in ``data/MYDATA/images/``.

Grid convention (matches the datasets in the paper): for each patch size the
centres form a regular grid, symmetric about the image centre, with an integer
spacing. Given ``n`` centres at spacing ``step``, the first centre sits at
``round((image_size - (n - 1) * step) / 2)`` and the rest follow at ``step``
intervals. Patches whose circle runs past the image border are kept and padded
with transparency, exactly as in Henderson & Hayes (2017).

Reference values, for calibrating a new set:

    P21-scegram  688x524   fine   r=53   n=12x10  step=58   120 patches
                           coarse r=123  n=8x6    step=97    48 patches  (168 total)
    HH25         1024x768  fine   r=43   n=20x15  step=51   300 patches
                           coarse r=102  n=12x9   step=85   108 patches  (408 total)

Running the script with those arguments reproduces the published coordinate
files for both datasets exactly.

Patch radii should be chosen in degrees of visual angle rather than pixels
(~3 deg fine, ~7 deg coarse in the original studies) so that a new dataset is
comparable to the published ones.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def centres(extent: int, n: int, step: int) -> list[int]:
    """``n`` grid centres at ``step`` spacing, centred within ``extent`` pixels."""
    span = (n - 1) * step
    first = int((extent - span) / 2 + 0.5)  # round half up
    return [first + i * step for i in range(n)]


def grid(imw: int, imh: int, radius: int, ncols: int, nrows: int, step: int):
    for yc in centres(imh, nrows, step):
        for xc in centres(imw, ncols, step):
            yield {"imw": imw, "imh": imh, "radius": radius, "xc": xc, "yc": yc}


class SizeSpec(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        w, _, h = values.partition("x")
        setattr(ns, self.dest, (int(w), int(h)))


def parse_scale(tokens: list[str]) -> dict:
    """Parse ``r=43 n=20x15 step=51`` into keyword values."""
    out = {}
    for tok in tokens:
        key, _, val = tok.partition("=")
        if key == "r":
            out["radius"] = int(val)
        elif key == "n":
            cols, _, rows = val.partition("x")
            out["ncols"], out["nrows"] = int(cols), int(rows)
        elif key == "step":
            out["step"] = int(val)
        else:
            raise SystemExit(f"unknown key in scale spec: {tok!r} (use r=, n=, step=)")
    missing = {"radius", "ncols", "nrows", "step"} - out.keys()
    if missing:
        raise SystemExit(f"scale spec is missing {sorted(missing)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="dataset name, e.g. MYDATA")
    ap.add_argument("--size", required=True, action=SizeSpec, metavar="WxH",
                    help="image size in pixels, e.g. 1024x768 (all images must share it)")
    ap.add_argument("--fine", nargs="+", required=True, metavar="K=V",
                    help="fine scale: r=43 n=20x15 step=51")
    ap.add_argument("--coarse", nargs="+", metavar="K=V",
                    help="coarse scale: r=102 n=12x9 step=85")
    ap.add_argument("--out", type=Path, help="output CSV (default data/{name}/{name}_patches_orig.csv)")
    args = ap.parse_args()

    imw, imh = args.size
    rows = list(grid(imw, imh, **parse_scale(args.fine)))
    if args.coarse:
        rows += list(grid(imw, imh, **parse_scale(args.coarse)))

    out = args.out or Path("data") / args.name / f"{args.name}_patches_orig.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["imw", "imh", "radius", "xc", "yc"])
        w.writeheader()
        w.writerows(rows)

    by_r: dict[int, int] = {}
    for r in rows:
        by_r[r["radius"]] = by_r.get(r["radius"], 0) + 1
    detail = ", ".join(f"r={r}: {n}" for r, n in sorted(by_r.items()))
    print(f"{out}: {len(rows)} patches per image ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
