"""
Multi-dataset publication plots.

Publication-ready figures that combine multiple datasets, with dataset identity
encoded by colour. Used by ``3_paper_plots.ipynb``.

Dependencies: numpy, scipy, matplotlib, pandas
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .map_comparison import SaliencyComparator

# Fine placement of the inline A/B/C panel letters in Fig 3. Horizontally they
# hang left of each panel's tight bbox, shifted right by LETTER_DX_MM. Vertically
# they sit on the same baseline as the dataset label (panel A's title), so the two
# read as one line; LETTER_DY_MM lifts them further above that baseline if needed.
LETTER_DX_MM = 1.5
LETTER_DY_MM = 0.0


class MultiDatasetPlotter:
    """
    Creates publication-ready figures that combine multiple datasets.

    Each dataset is registered with a color and optional display label.
    Datasets appear as separate positions on the x-axis, while map-level
    comparisons are encoded on the y-axis.
    """

    def __init__(self):
        """Initialize with an empty dataset registry."""
        self._datasets: Dict[str, dict] = {}
        self._dataset_order: List[str] = []

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def add_dataset(
        self,
        name: str,
        comparator: SaliencyComparator,
        color: str,
        label: str = None,
    ):
        """
        Register a dataset for plotting.

        Args:
            name: Internal dataset identifier.
            comparator: SaliencyComparator with loaded data and computed correlations.
            color: Dataset color for data points (any matplotlib-compatible color).
            label: Display label for x-axis (defaults to ``name``).
        """
        self._datasets[name] = {
            "comparator": comparator,
            "color": color,
            "label": label if label is not None else name,
        }
        if name not in self._dataset_order:
            self._dataset_order.append(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_correlation_values(
        self, dataset_name: str, map1: str, map2: str
    ) -> np.ndarray:
        """
        Return per-image correlation values between two maps for a dataset.

        Tries ``corr_df.loc[map1, map2]`` first; retries swapped (symmetric).
        """
        comp = self._datasets[dataset_name]["comparator"]
        values = []
        for corr_df in comp.all_correlations.values():
            if map1 in corr_df.index and map2 in corr_df.columns:
                val = corr_df.loc[map1, map2]
                if not np.isnan(val):
                    values.append(val)
        if not values:
            for corr_df in comp.all_correlations.values():
                if map2 in corr_df.index and map1 in corr_df.columns:
                    val = corr_df.loc[map2, map1]
                    if not np.isnan(val):
                        values.append(val)
        return np.array(values)

    @staticmethod
    def _safe_corr(corr_df: pd.DataFrame, map1: str, map2: str) -> float:
        """
        Extract one correlation value from a matrix, trying both orderings.
        Returns np.nan if neither ordering is found.
        """
        if map1 in corr_df.index and map2 in corr_df.columns:
            return corr_df.loc[map1, map2]
        if map2 in corr_df.index and map1 in corr_df.columns:
            return corr_df.loc[map2, map1]
        return np.nan

    @staticmethod
    def _annotate_stats(ax: plt.Axes, r: float, z: float, p_r: float, p_W: float,
                         fontsize: int = 8, show_r: bool = True):
        """Add r and/or z annotation to the lower-right corner of a scatter panel.
        Each statistic carries its own significance stars (p_r for r, p_W for z).
        Set show_r=False to annotate only z (e.g. for space-constrained panels).
        """
        def _stars(p):
            return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        text = f'r={r:.2f}{_stars(p_r)}\nz={z:.2f}{_stars(p_W)}' if show_r else f'z={z:.2f}{_stars(p_W)}'
        ax.text(
            0.97, 0.03, text,
            transform=ax.transAxes, fontsize=fontsize,
            ha='right', va='bottom', multialignment='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
        )

    # ------------------------------------------------------------------
    # Violin + scatter per dataset (generic helper)
    # ------------------------------------------------------------------

    def plot_violin_per_dataset(
        self,
        reference_map: str,
        test_map: str,
        datasets: List[str] = None,
        central_tendency: str = "median",
        figsize: Tuple[int, int] = (6, 5),
        ylabel: str = None,
        title: str = None,
        ylim: Tuple[float, float] = (0, 1),
        jitter_width: float = 0.04,
        dot_size: int = 40,
        save_path: str = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Figure 1: violin + scatter plot with one violin per dataset.

        X-axis: datasets
        Y-axis: per-image correlation between ``test_map`` and ``reference_map``

        Each dataset is represented by a gray violin body and colored scatter
        dots (with a larger colored central-tendency dot outlined in black).

        Args:
            reference_map: Map used as the reference in the correlation (e.g. ``'mm'``).
            test_map: Map whose correlation with ``reference_map`` is shown (e.g. ``'original'``).
            datasets: Ordered subset of dataset names to include.
                      Defaults to all registered datasets in add order.
            central_tendency: ``'median'`` (default) or ``'mean'``.
            figsize: Figure size ``(width, height)`` in inches.
            ylabel: Y-axis label.  Defaults to ``'Correlation with {reference_map}'``.
            title: Optional figure title.
            ylim: Y-axis limits ``(ymin, ymax)``.
            jitter_width: Standard deviation of the horizontal jitter for scatter dots.
            dot_size: Marker size for scatter dots.
            save_path: If provided, the figure is saved to this path.
            dpi: Resolution for the saved figure.

        Returns:
            The matplotlib ``Figure`` object.
        """
        if datasets is None:
            datasets = self._dataset_order

        all_values: List[np.ndarray] = []
        labels: List[str] = []
        colors: List[str] = []

        for ds_name in datasets:
            if ds_name not in self._datasets:
                raise ValueError(
                    f"Dataset '{ds_name}' not found. Call add_dataset() first."
                )
            vals = self._get_correlation_values(ds_name, reference_map, test_map)
            all_values.append(vals)
            labels.append(self._datasets[ds_name]["label"])
            colors.append(self._datasets[ds_name]["color"])

        fig, ax = plt.subplots(figsize=figsize)
        positions = list(range(len(datasets)))

        # Gray violin bodies
        parts = ax.violinplot(
            all_values,
            positions=positions,
            showmeans=(central_tendency == "mean"),
            showmedians=(central_tendency == "median"),
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("lightgray")
            pc.set_alpha(0.7)
        for partname in ["cmedians", "cmeans"]:
            if partname in parts:
                parts[partname].set_edgecolor("#A0A0A0")
                parts[partname].set_linewidth(1.5)

        # Colored scatter dots + central tendency
        rng = np.random.default_rng(42)
        for i, (vals, color) in enumerate(zip(all_values, colors)):
            x_jitter = rng.normal(i, jitter_width, len(vals))
            ax.scatter(
                x_jitter, vals,
                color=color, alpha=0.8, s=dot_size,
                edgecolors="white", linewidth=0.35, zorder=3,
            )
            central_val = (
                np.median(vals) if central_tendency == "median" else np.mean(vals)
            )
            ax.scatter(
                [i], [central_val],
                color=color, s=dot_size * 2,
                edgecolors="black", linewidth=1.0, zorder=5,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel(
            ylabel if ylabel is not None else f"Correlation with {reference_map}"
        )
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlabel("Dataset")

        if title:
            ax.set_title(title, fontsize=13, fontweight="bold")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    # ------------------------------------------------------------------
    # Fixation-correlation overview: A+B on top sub-row, C grid below
    # ------------------------------------------------------------------

    def plot_fixation_correlation_grid(
        self,
        datasets: List[str] = None,
        fixation_map: str = "fixation",
        test_map: str = "aimm",
        mm_map: str = "mm",
        saliency_maps: List[str] = None,
        map_labels: Dict[str, str] = None,
        central_tendency: str = "median",
        figsize: Tuple[int, int] = None,
        ylim: Tuple[float, float] = (0, 1),
        save_path: str = None,
        dpi: int = 300,
        show_stats_b: bool = False,
        show_stats_c: bool = False,
        stats_b: Optional[Dict] = None,
        stats_c: Optional[Dict] = None,
        base_fontsize: Optional[float] = None,
        tight_bbox: bool = True,
        dot_individual: float = 9.0,
        dot_group: float = 22.0,
        dot_scatter: float = 10.0,
        dot_scatter_small: float = 6.0,
        ab_height_mm: float = 25.0,
        b_width_mm: float = 25.0,
        margins_mm: Tuple[float, float, float, float] = (12.0, 1.5, 6.0, 6.0),
        block_gap_mm: float = 11.0,
        ab_c_gap_mm: float = 9.0,
        ab_gap_mm: float = 6.0,
        c_gap_mm: Tuple[float, float] = (6.0, 3.0),
        c_all_yticklabels: bool = True,
        letters_inline: bool = True,
        violin_xlabel: Optional[str] = "Map",
        violin_ylabel: str = "Correlation w/ fixations",
        verbose_layout: bool = False,
    ) -> plt.Figure:
        """
        Fixation-correlation overview — Figure 3.

        Each dataset occupies **two stacked sub-rows** so the DeepMeaning (DMM)
        column fits within a fixed journal width:

        * **Top sub-row**
            - **Panel A** (wider) – Violin + scatter: correlation with fixation
              on the y-axis, one violin per map (``test_map``, ``mm_map`` then
              each ``saliency_maps`` entry, e.g. AIMM, HMM, DMM, GBVS, AWS,
              DGIIE).
            - **Panel B** – Scatter: x = corr(``mm_map``, fix), y =
              corr(``test_map``, fix), one dot per image, with identity line.
        * **Bottom sub-row**
            - **Panel C** – 2 × ``n_sal`` grid of scatters: row 1 is
              ``test_map`` vs each ``saliency_maps`` entry, row 2 is ``mm_map``
              vs each, all measured via fixation correlation.  With
              ``saliency_maps=['dmm','gbvs','aws','deepgaze']`` this is a 2×4
              grid.

        The blocks stack vertically, one per dataset.

        Layout is specified in **absolute millimetres**, not ratios, because the
        B and C panels are ``aspect='equal'`` squares: with ratio-based spacing
        the squares shrink to whatever the sub-row height allows and the leftover
        cell width shows up as large horizontal gaps.  Sizing the panels directly
        keeps the C squares packed and makes the figure width a *consequence* of
        the panel sizes rather than an independent knob.

        Rule of thumb for a self-consistent figure (n_sal = 4):

            usable_w = fig_w_mm - margins_mm[0] - margins_mm[1]
            c_side   = (usable_w - 3 * c_gap_mm[0]) / 4
            block_h  = ab_height_mm + ab_c_gap_mm + 2 * c_side + c_gap_mm[1]
            fig_h_mm = n_ds * block_h + (n_ds - 1) * block_gap_mm
                       + margins_mm[2] + margins_mm[3]

        Pass ``verbose_layout=True`` to print the resulting panel sizes and see
        whether the C squares are width- or height-limited.

        Args mirror :meth:`plot_fixation_correlation_grid`, plus:
            ab_height_mm: Height of the top sub-row (panels A and B).
            b_width_mm: Width of panel B's cell; panel A takes the rest.
            margins_mm: (left, right, bottom, top) space reserved around the axes.
            block_gap_mm: Vertical gap between dataset blocks.
            ab_c_gap_mm: Vertical gap between the top sub-row and the C grid.
            ab_gap_mm: Horizontal gap between panels A and B.
            c_gap_mm: (horizontal, vertical) gaps inside the C grid.  The
                horizontal gap has to hold the y tick labels when
                ``c_all_yticklabels`` is on.
            c_all_yticklabels: Show y tick labels on every C column, not just the
                first (the y *axis label* stays on the first column either way).
            letters_inline: Place the A/B/C panel letters beside their axes (on the
                dataset label's baseline) instead of above them, so they don't eat
                into ``ab_c_gap_mm`` / ``block_gap_mm``.
            violin_xlabel: x-axis label for panel A; ``None`` drops it (the tick
                labels already name the maps) and frees vertical space.
            violin_ylabel: y-axis label for panel A.  Keep it short enough to fit
                inside ``ab_height_mm`` or it will run into the panel letter.
            verbose_layout: Print the derived panel geometry in mm.

        Returns:
            The matplotlib ``Figure`` object.
        """
        if datasets is None:
            datasets = self._dataset_order
        if saliency_maps is None:
            saliency_maps = ["dmm", "gbvs", "aws", "deepgaze"]
        if map_labels is None:
            map_labels = {
                fixation_map: "Fixations",
                test_map:     "AIMM",
                mm_map:       "HMM",
                "dmm":        "DMM",
                "gbvs":       "GBVS",
                "aws":        "AWS",
                "deepgaze":   "DGIIE",
            }

        n_ds  = len(datasets)
        n_sal = len(saliency_maps)   # typically 4 (dmm, gbvs, aws, deepgaze)

        if figsize is None:
            row_h = 5.5   # each dataset now spans two sub-rows (A+B, then C grid)
            figsize = (14, n_ds * row_h)

        # Font sizes derive from a single base so the whole figure can be
        # rescaled (e.g. to a fixed journal column width) in one place.
        base       = base_fontsize if base_fontsize is not None else 10.0
        fs_label   = base            # axis labels
        fs_tick    = base            # tick labels
        fs_title   = base            # per-row dataset title
        fs_stat_b  = base - 1.0      # panel B stat annotation
        fs_stat_c  = fs_stat_b       # panel C stat annotation
        fs_letter  = base + 1.5      # panel letters (A/B/C)

        fig = plt.figure(figsize=figsize)

        # ── Absolute (mm) layout → gridspec fractions ─────────────────────
        # gridspec's hspace/wspace are fractions of the *average* cell size, so
        # every gap is converted from mm once the cell sizes are known.
        MM = 25.4
        fig_w_mm, fig_h_mm = figsize[0] * MM, figsize[1] * MM
        m_left, m_right, m_bottom, m_top = margins_mm
        usable_w = fig_w_mm - m_left - m_right
        usable_h = fig_h_mm - m_bottom - m_top

        block_h = (usable_h - (n_ds - 1) * block_gap_mm) / n_ds
        h_top   = ab_height_mm
        h_bot   = block_h - ab_c_gap_mm - h_top
        if h_bot <= 0:
            raise ValueError(
                f"No room left for the C grid: block height {block_h:.1f} mm minus "
                f"ab_height_mm={ab_height_mm} and ab_c_gap_mm={ab_c_gap_mm}. "
                "Increase the figure height or shrink the top sub-row."
            )

        w_B = b_width_mm
        w_A = usable_w - ab_gap_mm - w_B

        c_gap_x, c_gap_y = c_gap_mm
        c_cell_w = (usable_w - (n_sal - 1) * c_gap_x) / n_sal
        c_cell_h = (h_bot - c_gap_y) / 2
        c_side   = min(c_cell_w, c_cell_h)   # panels are aspect='equal' squares

        if verbose_layout:
            print(f"[fig3 layout] figure        : {fig_w_mm:.1f} x {fig_h_mm:.1f} mm")
            print(f"[fig3 layout] block height  : {block_h:.1f} mm "
                  f"(A/B {h_top:.1f} + gap {ab_c_gap_mm:.1f} + C {h_bot:.1f})")
            print(f"[fig3 layout] panel A       : {w_A:.1f} x {h_top:.1f} mm")
            print(f"[fig3 layout] panel B square: {min(w_B, h_top):.1f} mm "
                  f"(cell {w_B:.1f} x {h_top:.1f})")
            print(f"[fig3 layout] C square      : {c_side:.1f} mm "
                  f"(cell {c_cell_w:.1f} x {c_cell_h:.1f}, "
                  f"{'width' if c_cell_w <= c_cell_h else 'height'}-limited); "
                  f"visual x-gap {c_gap_x + (c_cell_w - c_side):.1f} mm")

        # One outer block per dataset, stacked vertically. The top margin leaves
        # headroom for the A/B panel letters, drawn above the top sub-row.
        outer = gridspec.GridSpec(
            n_ds, 1,
            figure=fig,
            hspace=block_gap_mm / block_h,
            left=m_left / fig_w_mm,
            right=1.0 - m_right / fig_w_mm,
            bottom=m_bottom / fig_h_mm,
            top=1.0 - m_top / fig_h_mm,
        )

        rng = np.random.default_rng(42)
        # (ax_v, ax_s2, ax_inner_first, row_n) — filled per row
        label_placements = []

        for row_idx, ds_name in enumerate(datasets):
            ds       = self._datasets[ds_name]
            comp     = ds["comparator"]
            color    = ds["color"]
            ds_label = ds["label"]

            # Two stacked sub-rows within this dataset's block.
            block = gridspec.GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=outer[row_idx],
                height_ratios=[h_top, h_bot],
                hspace=ab_c_gap_mm / ((h_top + h_bot) / 2),
            )
            # Top sub-row: A (wider) + B.
            top = gridspec.GridSpecFromSubplotSpec(
                1, 2,
                subplot_spec=block[0],
                width_ratios=[w_A, w_B],
                wspace=ab_gap_mm / ((w_A + w_B) / 2),
            )

            # ── Panel A: violin corr vs fixation ──────────────────────────
            ax_v = fig.add_subplot(top[0, 0])

            violin_maps    = [m for m in [test_map, mm_map] + saliency_maps]
            violin_data    = []
            violin_xlabels = []

            for m in violin_maps:
                vals = self._get_correlation_values(ds_name, fixation_map, m)
                violin_data.append(vals)
                violin_xlabels.append(map_labels.get(m, m))

            non_empty = [i for i, v in enumerate(violin_data) if len(v) > 0]

            if non_empty:
                positions = list(range(len(violin_maps)))
                parts = ax_v.violinplot(
                    violin_data,
                    positions=positions,
                    showmeans=(central_tendency == "mean"),
                    showmedians=(central_tendency == "median"),
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor("lightgray")
                    pc.set_alpha(0.7)
                for pn in ["cmedians", "cmeans"]:
                    if pn in parts:
                        parts[pn].set_edgecolor("#A0A0A0")
                        parts[pn].set_linewidth(0.3)

                for i, vals in enumerate(violin_data):
                    if len(vals) == 0:
                        continue
                    xj = rng.normal(i, 0.04, len(vals))
                    ax_v.scatter(xj, vals, color=color, alpha=0.7, s=dot_individual,
                                 edgecolors="white", linewidth=0.35, zorder=3)
                    cv = np.median(vals) if central_tendency == "median" else np.mean(vals)
                    ax_v.scatter([i], [cv], color=color, s=dot_group,
                                 edgecolors="black", linewidth=0.5, zorder=5)

            ax_v.set_xticks(range(len(violin_maps)))
            ax_v.set_xticklabels(violin_xlabels, rotation=0, fontsize=fs_tick)
            # Half a slot of padding so the outer violin bodies never touch the
            # spines (default 5% margins are too tight for wide violins).
            ax_v.set_xlim(-0.55, len(violin_maps) - 0.45)
            ax_v.set_ylim(*ylim)
            ax_v.set_ylabel(violin_ylabel, fontsize=fs_label)
            if violin_xlabel:
                ax_v.set_xlabel(violin_xlabel, fontsize=fs_label)
            ax_v.tick_params(axis='y', labelsize=fs_tick)
            ax_v.grid(True, alpha=0.3, axis="y")
            ax_v.set_title(ds_label, fontsize=fs_title, fontweight="bold", loc="left", pad=4)

            # ── Panel B: scatter HMM vs AIMM (both vs fixation) ───────────
            ax_s2 = fig.add_subplot(top[0, 1])
            self._draw_scatter_vs_fix(
                ax_s2, comp, mm_map, test_map, fixation_map,
                xlabel=f"{map_labels.get(mm_map, mm_map)}",
                ylabel=f"{map_labels.get(test_map, test_map)}",
                color=color,
                label_fontsize=fs_label,
                tick_fontsize=fs_tick,
                dot_size=dot_scatter,
            )
            # Same tick steps on both axes of B and on panel A's y-axis, so the
            # auto-locator can't switch to 0.25 steps when B grows.
            ax_s2.set_xticks(np.arange(0, 1.01, 0.2))
            ax_s2.set_yticks(np.arange(0, 1.01, 0.2))
            if show_stats_b and stats_b is not None and ds_name in stats_b:
                s = stats_b[ds_name]
                self._annotate_stats(ax_s2, s['r'], s['z'], s['p_r'], s['p_W'], fontsize=fs_stat_b)

            # ── Panel C: 2 × n_sal scatter grid (bottom sub-row) ──────────
            ax_inner_first = None
            inner = gridspec.GridSpecFromSubplotSpec(
                2, n_sal,
                subplot_spec=block[1],
                hspace=c_gap_y / c_cell_h,
                wspace=c_gap_x / c_cell_w,
            )

            row_maps = [test_map, mm_map]
            for r, ref_map in enumerate(row_maps):
                for c, sal_map in enumerate(saliency_maps):
                    ax_sc = fig.add_subplot(inner[r, c])
                    if ax_inner_first is None:
                        ax_inner_first = ax_sc
                    show_xlabel = (r == len(row_maps) - 1)
                    show_ylabel = (c == 0)
                    self._draw_scatter_vs_fix(
                        ax_sc, comp, sal_map, ref_map, fixation_map,
                        xlabel=map_labels.get(sal_map, sal_map),
                        ylabel=map_labels.get(ref_map, ref_map),
                        color=color,
                        label_fontsize=fs_label,
                        tick_fontsize=fs_tick,
                        dot_size=dot_scatter_small,
                    )
                    ax_sc.set_xticks([0.0, 0.5, 1.0])
                    ax_sc.set_yticks([0.0, 0.5, 1.0])
                    ax_sc.set_xticklabels(["", "0.5", "1.0"])  # hide 0.0 on x-axis
                    ax_sc.set_yticklabels(["", "0.5", "1.0"])  # hide 0.0 on y-axis
                    if not show_xlabel:
                        ax_sc.set_xlabel("")
                        ax_sc.tick_params(labelbottom=False)
                    if not show_ylabel:
                        ax_sc.set_ylabel("")
                        if not c_all_yticklabels:
                            ax_sc.tick_params(labelleft=False)
                    if show_stats_c and stats_c is not None:
                        key = (ds_name, ref_map, sal_map)
                        if key in stats_c:
                            s = stats_c[key]
                            self._annotate_stats(ax_sc, s['r'], s['z'], s['p_r'], s['p_W'],
                                                  fontsize=fs_stat_c, show_r=False)

            label_placements.append((ax_v, ax_s2, ax_inner_first, row_idx + 1))

        # Place panel labels using tight bounding boxes so each label sits at
        # the left edge of its panel's full extent (including y-axis label).
        # With ``letters_inline`` they sit in the strip left of the y-axis label,
        # on the dataset label's baseline, so they cost no vertical space and the
        # gaps between the sub-rows stay real whitespace.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_inv = fig.transFigure.inverted()

        label_kw = dict(fontsize=fs_letter, fontweight="bold",
                        va=("baseline" if letters_inline else "bottom"),
                        ha=("right" if letters_inline else "left"),
                        clip_on=False, fontfamily='Arial')

        # Inline letters hang just left of the panel's tight bbox (i.e. left of
        # its y-axis label and tick labels), which is why they need ~6 mm of
        # clearance in margins_mm[0] and in ab_gap_mm.
        pad_px  = (1.0 / MM) * fig.dpi
        letter_x = (lambda ax: fig_inv.transform(
            [ax.get_tightbbox(renderer).x0 - (pad_px if letters_inline else 0), 0])[0])

        fig_w_mm, fig_h_mm = (s * MM for s in fig.get_size_inches())
        dx_letter = (LETTER_DX_MM / fig_w_mm) if letters_inline else 0.0
        dy_letter = (LETTER_DY_MM / fig_h_mm) if letters_inline else 0.0

        def _title_baseline_offset(ax):
            """Height of ``ax``'s title baseline above the axes top, in fig fraction.

            Read back from the title matplotlib just laid out, so the letters keep
            tracking the dataset labels if the title pad changes. The pad lives in
            the title's *transform* (not its position), hence the transform round
            trip; titles are drawn ``va="baseline"``, so the letters share it.
            """
            title_y = fig_inv.transform(
                ax.title.get_transform().transform(ax.title.get_position()))[1]
            return title_y - ax.get_position().y1

        for ax_v, ax_s2, ax_c_first, row_n in label_placements:
            # A and B share the top sub-row -> common label height. Inline letters
            # sit on the dataset label's baseline; panel B and the C grid carry no
            # title, so they reuse the offset measured on panel A.
            ax_v_pos = ax_v.get_position()
            ax_v_h   = ax_v_pos.y1 - ax_v_pos.y0
            title_dy = _title_baseline_offset(ax_v) if letters_inline else 0.06 * ax_v_h
            label_y_top = ax_v_pos.y1 + title_dy + dy_letter

            fig.text(letter_x(ax_v)  + dx_letter, label_y_top, f"A{row_n}", **label_kw)
            fig.text(letter_x(ax_s2) + dx_letter, label_y_top, f"B{row_n}", **label_kw)

            # C sits on the lower sub-row -> its own label height.
            if ax_c_first is not None:
                c_pos = ax_c_first.get_position()
                c_h   = c_pos.y1 - c_pos.y0
                label_y_c = c_pos.y1 + (title_dy if letters_inline else 0.10 * c_h)
                label_y_c += dy_letter
                fig.text(letter_x(ax_c_first) + dx_letter, label_y_c, f"C{row_n}", **label_kw)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=dpi,
                        bbox_inches=("tight" if tight_bbox else None))

        return fig

    # ------------------------------------------------------------------
    # Scatter helper
    # ------------------------------------------------------------------

    def _draw_scatter_vs_fix(
        self,
        ax: plt.Axes,
        comp: SaliencyComparator,
        map1: str,
        map2: str,
        fixation_map: str,
        xlabel: str = "",
        ylabel: str = "",
        color: str = "gray",
        label_fontsize: int = 8,
        tick_fontsize: int = 7,
        dot_size: int = 20,
    ):
        """
        Draw a scatter plot on ``ax`` where:
          x = corr(map1, fixation) per image
          y = corr(map2, fixation) per image
        with an identity line (y = x).

        Leaves the axes empty (with "N/A" label) if either map is missing.
        """
        x_vals, y_vals = [], []
        for corr_df in comp.all_correlations.values():
            x = self._safe_corr(corr_df, fixation_map, map1)
            y = self._safe_corr(corr_df, fixation_map, map2)
            if not (np.isnan(x) or np.isnan(y)):
                x_vals.append(x)
                y_vals.append(y)

        if x_vals:
            ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=dot_size,
                       edgecolors="white", linewidth=0.35, zorder=3)
            ax.plot([0, 1], [0, 1], color="#A0A0A0", linewidth=0.3, zorder=2)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=2)
        ax.tick_params(labelsize=tick_fontsize)
        ax.grid(True, alpha=0.25)

