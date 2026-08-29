import glob
import os
import re
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image
import scipy.io
from scipy.ndimage import zoom


def reconstruct_images(df, metric_column, output_dir, model, prompt_id, shape='round', verbose=True):
    """
    Reconstruct meaningfulness maps from patch scores (2025 method).

    For each image:
      1. Per patch size: paint each patch's score into a pixel grid and
         average values in overlapping regions (no smoothing or interpolation).
      2. Average the per-size maps pixel-wise using nanmean (pixels not covered
         by any patch in a given size are excluded from that size's contribution).

    Outputs per image (saved to output_dir):
      - <image_name>.npy   raw float32 map
      - <image_name>.png   grayscale preview (min-max normalised to 0-255)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'image_path', 'model', 'prompt_id', and metric_column.
    metric_column : str
        Column with the score to reconstruct.
    output_dir : str
        Directory where outputs are saved.
    model : str
        Model name used to filter df.
    prompt_id : str
        Prompt ID used to filter df.
    shape : {'round', 'square'}
        Patch shape. Default 'round'.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    dict
        {image_name: combined_float32_array}
    """
    os.makedirs(output_dir, exist_ok=True)

    filtered_df = df[(df['model'] == model) & (df['prompt_id'] == prompt_id)].copy()

    if len(filtered_df) == 0:
        if verbose:
            print(f"No patches found for model '{model}' and prompt_id '{prompt_id}'")
        return {}

    if verbose:
        print(f"Processing {len(filtered_df)} patches for model '{model}', prompt '{prompt_id}'")

    # Parse patch metadata from filenames
    patch_info_list = []
    for _, row in filtered_df.iterrows():
        info = _extract_patch_info(row['image_path'])
        if info:
            info['metric_value'] = row[metric_column]
            patch_info_list.append(info)
        elif verbose:
            print(f"Warning: could not parse filename: {row['image_path']}")

    if not patch_info_list:
        if verbose:
            print("No valid patch filenames found.")
        return {}

    patch_df = pd.DataFrame(patch_info_list)
    original_images = patch_df['original_image'].unique()

    if verbose:
        print(f"Found {len(original_images)} unique images")

    results = {}

    for img_name in original_images:
        if verbose:
            print(f"Processing {img_name}...")

        img_patches = patch_df[patch_df['original_image'] == img_name]
        radii = sorted(img_patches['radius'].unique())

        first = img_patches.iloc[0]
        orig_w, orig_h = int(first['width']), int(first['height'])

        if verbose:
            print(f"  Image size: {orig_w}x{orig_h}, radii: {radii}")

        per_size_maps = []

        for radius in radii:
            r_patches = img_patches[img_patches['radius'] == radius]

            accumulated = np.zeros((orig_h, orig_w), dtype=np.float64)
            count = np.zeros((orig_h, orig_w), dtype=np.float64)

            if shape == 'round':
                gy, gx = np.ogrid[-radius:radius, -radius:radius]
                circ_mask = gx**2 + gy**2 <= radius**2

            for _, patch in r_patches.iterrows():
                cx, cy = int(patch['x']), int(patch['y'])
                val = float(patch['metric_value'])

                src_l = cx - radius
                src_t = cy - radius
                src_r = cx + radius
                src_b = cy + radius

                cl = max(0, src_l)
                ct = max(0, src_t)
                cr = min(orig_w, src_r)
                cb = min(orig_h, src_b)

                if cl >= cr or ct >= cb:
                    continue

                ph = cb - ct
                pw = cr - cl

                if shape == 'round':
                    my0 = ct - src_t
                    mx0 = cl - src_l
                    patch_mask = circ_mask[my0:my0 + ph, mx0:mx0 + pw]
                    if patch_mask.shape != (ph, pw):
                        patch_mask = np.ones((ph, pw), dtype=bool)
                else:
                    patch_mask = np.ones((ph, pw), dtype=bool)

                accumulated[ct:cb, cl:cr][patch_mask] += val
                count[ct:cb, cl:cr][patch_mask] += 1

            # NaN for pixels not covered by any patch of this size
            size_map = np.where(count > 0, accumulated / count, np.nan)
            per_size_maps.append(size_map)

            if verbose:
                covered = np.sum(count > 0)
                print(f"  Radius {radius}: {len(r_patches)} patches, "
                      f"{covered}/{orig_w * orig_h} pixels covered")

        # Average across patch sizes (nanmean ignores sizes that didn't cover a pixel)
        combined = np.nanmean(np.stack(per_size_maps, axis=0), axis=0).astype(np.float32)

        # --- Save outputs ---

        # .npy
        npy_path = os.path.join(output_dir, f"{img_name}.npy")
        np.save(npy_path, combined)

        # .png  (grayscale preview, min-max normalised)
        valid = combined[~np.isnan(combined)]
        if valid.size > 0 and valid.max() > valid.min():
            preview = (combined - valid.min()) / (valid.max() - valid.min()) * 255.0
            preview = np.nan_to_num(preview, nan=0.0).astype(np.uint8)
        else:
            preview = np.zeros((orig_h, orig_w), dtype=np.uint8)

        png_path = os.path.join(output_dir, f"{img_name}.png")
        Image.fromarray(preview, mode='L').save(png_path)

        results[img_name] = combined

        if verbose:
            print(f"  Saved {img_name}.npy / .png  "
                  f"(values {np.nanmin(combined):.3f}–{np.nanmax(combined):.3f})")

    return results


def load_cb_template(cb_template_path='data/templates_cb/centerbias_gbvs.txt'):
    """Load the GBVS center-bias weight map used by ``apply_center_bias``.

    Two interchangeable encodings of the same 32x32 template are accepted:

    ``centerbias_gbvs.txt``
        Comma-separated, values in [0, 0.5]. This is the one tracked in the
        repository, and it holds ``0.5 * (1 - invCenterBias)``.
    ``invCenterBias.mat``
        The GBVS artifact from ``gbvs-master/util/``, values in [0, 1]. Not
        redistributed here; supported so an existing copy still works.

    Both return the same weight map, ``0.5 + 0.5 * (1 - invCenterBias)``, which
    is what mfiles/apply_cb.m applies. Weights run from 0.5 at the corners to
    1.0 at the centre.
    """
    if str(cb_template_path).lower().endswith('.mat'):
        inv_center_bias = scipy.io.loadmat(cb_template_path)['invCenterBias'].astype(float)
        cb = inv_center_bias * (-1) + 1          # invert -> [0, 1]
        return 0.5 + 0.5 * cb
    # The .txt already carries the 0.5 factor, so it only needs the offset.
    return 0.5 + np.loadtxt(cb_template_path, delimiter=',').astype(float)


def apply_center_bias(input_dir, output_dir, cb_template_path='data/templates_cb/centerbias_gbvs.txt'):
    """
    Apply center bias to PNG maps in input_dir and save results to output_dir.

    Mirrors the MATLAB script mfiles/apply_cb.m:
      CB = 0.5 + 0.5 * (1 - invCenterBias)
      master_map = master_map .* CB   (bicubic resize to match each image)
      master_map = mat2gray(master_map)

    Parameters
    ----------
    input_dir : str
        Directory containing source PNG maps (e.g. maps_combined_raw/).
    output_dir : str
        Directory where center-biased PNGs are saved (e.g. maps_combined_cb/).
    cb_template_path : str
        Path to the center-bias template; see ``load_cb_template``.
    """
    CB_template = load_cb_template(cb_template_path)

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    print(f"Applying center bias to {len(files)} maps...")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    for fpath in files:
        img = np.array(Image.open(fpath)).astype(float)
        h, w = img.shape[:2]

        # Resize CB to match image dimensions (bicubic, like MATLAB imresize default)
        CB = zoom(CB_template, (h / CB_template.shape[0], w / CB_template.shape[1]), order=3)

        # Apply CB and normalize to [0, 1] (mat2gray equivalent)
        master_map = img * CB
        vmin, vmax = master_map.min(), master_map.max()
        if vmax > vmin:
            master_map = (master_map - vmin) / (vmax - vmin)

        fname = os.path.basename(fpath)
        Image.fromarray((master_map * 255).astype(np.uint8)).save(os.path.join(output_dir, fname))

    print(f"Done.")


def _extract_patch_info(patch_path):
    """Parse patch metadata from filename: {name}_w{W}_h{H}_x{X}_y{Y}_r{R}.ext"""
    # Normalize Windows backslashes then take basename — handles paths from any OS.
    filename = PurePosixPath(patch_path.replace('\\', '/')).stem
    pattern = r'^(.+?)_w(\d+)_h(\d+)_x(\d+)_y(\d+)_r(\d+)$'
    m = re.match(pattern, filename)
    if m:
        return {
            'original_image': m.group(1),
            'width': int(m.group(2)),
            'height': int(m.group(3)),
            'x': int(m.group(4)),
            'y': int(m.group(5)),
            'radius': int(m.group(6)),
        }
    return None
