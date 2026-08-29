"""
Saliency Map Comparison Module

A comprehensive module for comparing different types of saliency maps with ground truth fixation data.
Supports correlation analysis, visualization, and statistical comparison across multiple images.

Author: Assistant
Dependencies: numpy, scipy, matplotlib, seaborn, opencv-python, pandas, pathlib
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter
import scipy.io
import cv2
from skimage import exposure, io, transform

from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
from itertools import combinations
import matplotlib.patches as mpatches


class StatisticalComparator:
    """Handles statistical comparisons between saliency map correlations."""
    
    def __init__(self, all_correlations: Dict[str, pd.DataFrame]):
        """
        Initialize with correlation data from all images.
        
        Args:
            all_correlations: Dictionary of image_name -> correlation matrix
        """
        self.all_correlations = all_correlations
        self.correlation_vectors = self._extract_correlation_vectors()
    
    def _extract_correlation_vectors(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Extract correlation vectors for each map pair across all images.
        
        Returns:
            Dictionary with structure: {map1: {map2: [correlations across images]}}
        """
        if not self.all_correlations:
            raise ValueError("No correlation data provided")
        
        # Get map types from first correlation matrix
        first_corr = next(iter(self.all_correlations.values()))
        map_types = list(first_corr.index)
        
        # Initialize storage
        correlation_vectors = {}
        for map1 in map_types:
            correlation_vectors[map1] = {}
            for map2 in map_types:
                correlation_vectors[map1][map2] = []
        
        # Extract correlations across all images
        for img_name, corr_matrix in self.all_correlations.items():
            for map1 in map_types:
                for map2 in map_types:
                    if map1 in corr_matrix.index and map2 in corr_matrix.columns:
                        corr_val = corr_matrix.loc[map1, map2]
                        if not np.isnan(corr_val):
                            correlation_vectors[map1][map2].append(corr_val)
        
        return correlation_vectors
    
    def compare_map_pairs(self, reference_map: str, test_maps: List[str], 
                         test_type: str = 'paired_t', alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare correlations of different maps against a reference map.
        
        Args:
            reference_map: Name of reference map (e.g., 'fixation')
            test_maps: List of maps to compare against reference
            test_type: 'paired_t', 'wilcoxon', or 'mannwhitney'
            alpha: Significance level
        
        Returns:
            DataFrame with comparison results
        """
        if reference_map not in self.correlation_vectors:
            raise ValueError(f"Reference map '{reference_map}' not found")
        
        results = []
        
        # Get all pairwise combinations of test maps
        for map1, map2 in combinations(test_maps, 2):
            if map1 not in self.correlation_vectors or map2 not in self.correlation_vectors:
                continue
            
            # Get correlation vectors for each map with reference
            corr1_ref = self.correlation_vectors[map1][reference_map]
            corr2_ref = self.correlation_vectors[map2][reference_map]
            
            # Ensure we have data for both
            if len(corr1_ref) == 0 or len(corr2_ref) == 0:
                continue
            
            # Align data (only use images where we have both correlations)
            common_length = min(len(corr1_ref), len(corr2_ref))
            if common_length < 3:  # Need minimum samples
                continue
            
            corr1_ref = corr1_ref[:common_length]
            corr2_ref = corr2_ref[:common_length]
            
            # Perform statistical test
            try:
                if test_type == 'paired_t':
                    statistic, p_value = ttest_rel(corr1_ref, corr2_ref)
                    test_name = "Paired t-test"
                elif test_type == 'wilcoxon':
                    statistic, p_value = wilcoxon(corr1_ref, corr2_ref)
                    test_name = "Wilcoxon signed-rank test"
                elif test_type == 'mannwhitney':
                    statistic, p_value = mannwhitneyu(corr1_ref, corr2_ref, alternative='two-sided')
                    test_name = "Mann-Whitney U test"
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                
                # Calculate effect size (Cohen's d for t-test, r for non-parametric)
                if test_type == 'paired_t':
                    diff = np.array(corr1_ref) - np.array(corr2_ref)
                    effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                    effect_size_name = "Cohen's d"
                else:
                    effect_size = statistic / np.sqrt(common_length)
                    effect_size_name = "Effect size r"
                
                results.append({
                    'map1': map1,
                    'map2': map2,
                    'reference': reference_map,
                    'mean_corr1': np.mean(corr1_ref),
                    'mean_corr2': np.mean(corr2_ref),
                    'mean_diff': np.mean(corr1_ref) - np.mean(corr2_ref),
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': effect_size,
                    'effect_size_name': effect_size_name,
                    'test_type': test_name,
                    'n_samples': common_length
                })
                
            except Exception as e:
                warnings.warn(f"Statistical test failed for {map1} vs {map2}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_pairwise_comparison(self, reference_map: str, test_maps: List[str],
                               comparison_results: pd.DataFrame = None,
                               test_type: str = 'paired_t', figsize: Tuple[int, int] = (15, 6),
                               save_path: str = None):
        """
        Create visualization of pairwise statistical comparisons.
        
        Args:
            reference_map: Name of reference map
            test_maps: List of maps to compare
            comparison_results: Pre-computed results (optional)
            test_type: Statistical test type
            figsize: Figure size
            save_path: Optional path to save figure
        """
        # Compute results if not provided
        if comparison_results is None:
            comparison_results = self.compare_map_pairs(reference_map, test_maps, test_type)
        
        if comparison_results.empty:
            print("No valid comparisons found")
            return
        
        # Create figure with subplots - 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Distribution comparison (violin plot) - moved to first position
        violin_data = []
        violin_labels = []
        for test_map in test_maps:
            if test_map in self.correlation_vectors and reference_map in self.correlation_vectors[test_map]:
                corrs = self.correlation_vectors[test_map][reference_map]
                if corrs:
                    violin_data.append(corrs)
                    violin_labels.append(test_map)
        
        if violin_data:
            parts = ax1.violinplot(violin_data, positions=range(len(violin_labels)), showmeans=True, showmedians=False)
            ax1.set_xticks(range(len(violin_labels)))
            ax1.set_xticklabels(violin_labels, rotation=45)
            ax1.set_ylabel(f'Correlation with {reference_map}')
            ax1.set_title('Distribution of Correlations')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Color the violin plots
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
        
        # 2. Pairwise comparison with individual lines and means
        if not comparison_results.empty:
            n_comparisons = len(comparison_results)
            x_positions = np.arange(n_comparisons)
            
            # For each comparison, plot individual image pairs and mean
            for i, (_, row) in enumerate(comparison_results.iterrows()):
                map1, map2 = row['map1'], row['map2']
                
                # Get correlation vectors for this pair
                corr1_ref = self.correlation_vectors[map1][reference_map]
                corr2_ref = self.correlation_vectors[map2][reference_map]
                
                # Ensure same length
                common_length = min(len(corr1_ref), len(corr2_ref))
                corr1_ref = corr1_ref[:common_length]
                corr2_ref = corr2_ref[:common_length]
                
                # Plot individual image lines (thin, light gray)
                for j in range(common_length):
                    ax2.plot([i-0.15, i+0.15], [corr1_ref[j], corr2_ref[j]], 
                            color='lightgray', alpha=0.6, linewidth=0.8, zorder=1)
                
                # Plot means as thick line
                mean1, mean2 = np.mean(corr1_ref), np.mean(corr2_ref)
                ax2.plot([i-0.15, i+0.15], [mean1, mean2], 
                        color='black', linewidth=3, zorder=3)
                
                # Add points for means
                ax2.scatter([i-0.15, i+0.15], [mean1, mean2], 
                           color=['steelblue', 'orange'], s=80, zorder=4, edgecolor='black', linewidth=1)
                
                # Add significance asterisks
                if row['significant']:
                    y_max = max(max(corr1_ref), max(corr2_ref))
                    # y_text = y_max + 0.05
                    y_text = 0.35  # Adjusted for better visibility

                    # Determine significance level
                    p_val = row['p_value']
                    if p_val < 0.001:
                        sig_text = '***'
                    elif p_val < 0.01:
                        sig_text = '**'
                    elif p_val < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    ax2.text(i, y_text, sig_text, ha='center', va='bottom', 
                            fontweight='bold', color='red', fontsize=12)
                    
                    # Add horizontal line for significance
                    ax2.plot([i-0.2, i+0.2], [y_text-0.02, y_text-0.02], 
                            color='red', linewidth=1)
            
            # Customize the plot
            ax2.set_ylim(0, 1)
            comparison_labels = [f"{row['map1']}\nvs\n{row['map2']}" for _, row in comparison_results.iterrows()]
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(comparison_labels, fontsize=10)
            ax2.set_ylabel(f'Correlation with {reference_map}')
            ax2.set_title('Pairwise Statistical Comparisons')
            ax2.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                # Line2D([0], [0], color='lightgray', alpha=0.6, linewidth=0.8, label='Individual images'),
                # Line2D([0], [0], color='black', linewidth=3, label='Mean values'),
                plt.scatter([], [], color='steelblue', s=80, label='Map 1', edgecolor='black'),
                plt.scatter([], [], color='orange', s=80, label='Map 2', edgecolor='black')
            ]
            ax2.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0))
        
        plt.suptitle(f'Statistical Comparison: Correlations with {reference_map.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def create_summary_table(self, reference_map: str, test_maps: List[str],
                           test_type: str = 'paired_t') -> pd.DataFrame:
        """
        Create a comprehensive summary table of all comparisons.
        
        Args:
            reference_map: Reference map name
            test_maps: List of test maps
            test_type: Statistical test type
        
        Returns:
            Formatted summary DataFrame
        """
        results = self.compare_map_pairs(reference_map, test_maps, test_type)
        
        if results.empty:
            return pd.DataFrame()
        
        # Create formatted summary
        summary = results.copy()
        summary['mean_diff_formatted'] = summary['mean_diff'].apply(lambda x: f"{x:+.4f}")
        summary['p_value_formatted'] = summary['p_value'].apply(
            lambda x: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}"
        )
        summary['significance'] = summary['p_value'].apply(
            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns'
        )
        
        # Select and reorder columns
        display_cols = ['map1', 'map2', 'mean_corr1', 'mean_corr2', 'mean_diff_formatted', 
                       'p_value_formatted', 'significance', 'effect_size', 'n_samples']
        
        summary_display = summary[display_cols].copy()
        summary_display.columns = ['Map 1', 'Map 2', 'Mean Corr 1', 'Mean Corr 2', 
                                  'Difference', 'p-value', 'Sig.', 'Effect Size', 'N']
        
        return summary_display


class SaliencyMapLoader:
    """Handles loading and organizing saliency maps from different directories."""
    
    def __init__(self, image_dir: str, map_configs: Dict[str, str]):
        """
        Initialize the loader with directory configurations.
        
        Args:
            image_dir: Directory containing original images
            map_configs: Dictionary mapping map type names to their directories
        """
        self.image_dir = Path(image_dir)
        self.map_configs = {name: Path(path) for name, path in map_configs.items()}
        self.image_data = {}
        self.available_images = set()
        
        self._validate_directories()
    
    def _validate_directories(self):
        """Validate that all specified directories exist."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        for map_type, map_dir in self.map_configs.items():
            if not map_dir.exists():
                raise FileNotFoundError(f"Map directory for '{map_type}' not found: {map_dir}")
    
    # Formats tried in order of preference when loading maps.
    _MAP_FORMATS = ('.npy', '.mat', '.png')

    def _get_available_stems(self, directory: Path) -> set:
        """Return stems of all files in directory that have a supported map format."""
        stems = set()
        for ext in self._MAP_FORMATS:
            for f in directory.glob(f'*{ext}'):
                stems.add(f.stem)
        return stems

    def _find_common_images(self) -> List[str]:
        """Find image stems that exist in all specified directories."""
        all_stems = {}

        # Original images are always PNG
        all_stems['__image__'] = {f.stem for f in self.image_dir.glob('*.png')}

        # Map directories may use any supported format
        for map_type, map_dir in self.map_configs.items():
            all_stems[map_type] = self._get_available_stems(map_dir)

        common_images = set.intersection(*all_stems.values())

        if not common_images:
            raise ValueError("No common images found across all directories")

        return sorted(common_images)

    def _load_map(self, directory: Path, stem: str) -> np.ndarray:
        """
        Load a map for *stem* from *directory*.

        Format priority: .npy (raw floats) → .mat (raw floats, key='map') → .png
        PNG files are normalised to [0, 1]; .npy / .mat are returned as-is.
        """
        npy_path = directory / f"{stem}.npy"
        if npy_path.exists():
            return np.load(npy_path).astype(np.float64)

        mat_path = directory / f"{stem}.mat"
        if mat_path.exists():
            data = scipy.io.loadmat(mat_path)
            # Use the first non-metadata key (metadata keys start with '_')
            key = next(k for k in data if not k.startswith('_'))
            return np.squeeze(data[key]).astype(np.float64)

        png_path = directory / f"{stem}.png"
        if png_path.exists():
            arr = io.imread(png_path, as_gray=True).astype(np.float64)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr

        raise FileNotFoundError(
            f"No map found for '{stem}' in {directory} "
            f"(tried .npy, .mat, .png)"
        )
    
    def load_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all saliency maps and organize by image name.
        
        Returns:
            Dictionary with structure: {image_name: {map_type: array}}
        """
        common_images = self._find_common_images()
        self.available_images = set(common_images)
        
        print(f"Loading {len(common_images)} common images...")
        
        for img_name in common_images:
            self.image_data[img_name] = {}

            # Original images are always PNG
            orig_path = self.image_dir / f"{img_name}.png"
            self.image_data[img_name]['__image__'] = io.imread(orig_path)

            # Load each map type using the best available format
            for map_type, map_dir in self.map_configs.items():
                try:
                    map_array = self._load_map(map_dir, img_name)
                    self.image_data[img_name][map_type] = map_array
                except Exception as e:
                    warnings.warn(f"Failed to load {map_type} for {img_name}: {e}")
                    continue
        
        print(f"Successfully loaded data for {len(self.image_data)} images")
        return self.image_data


class CorrelationComputer:
    """Computes various correlation metrics between saliency maps."""
    
    @staticmethod
    def smooth_map(arr: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian smooth then rescale to the original value range.

        Replicates the Henderson & Hayes (2025) DeepMeaning convention:
        gaussian_filter(sigma) followed by norm_range back to [orig_min, orig_max].
        """
        orig_min, orig_max = arr.min(), arr.max()
        smoothed = gaussian_filter(arr, sigma=sigma)
        if orig_max > orig_min:
            s_min, s_max = smoothed.min(), smoothed.max()
            if s_max > s_min:
                smoothed = (smoothed - s_min) / (s_max - s_min) * (orig_max - orig_min) + orig_min
        return smoothed

    @staticmethod
    def apply_histogram_matching(maps: Dict[str, np.ndarray],
                               reference_type: str = None,
                               exclude_from_matching: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Apply histogram matching to all maps.

        Args:
            maps: Dictionary of map_type -> array
            reference_type: Which map to use as reference (None = use first map)
            exclude_from_matching: Map names to leave unmatched (keep their original
                histogram) — e.g. ['fixation'] to exclude fixations when matching to
                deepgaze. The reference map is always left unchanged.

        Returns:
            Dictionary of histogram-matched maps
        """
        map_types = list(maps.keys())
        if '__image__' in map_types:
            map_types.remove('__image__')  # Don't match original image

        if len(map_types) < 2:
            return maps

        # Choose reference
        if reference_type is None or reference_type not in map_types:
            reference_type = map_types[0]

        exclude_from_matching = exclude_from_matching or []

        reference_map = maps[reference_type]
        matched_maps = maps.copy()

        for map_type in map_types:
            if map_type != reference_type and map_type not in exclude_from_matching:
                try:
                    matched_maps[map_type] = exposure.match_histograms(
                        maps[map_type], reference_map
                    )
                except Exception as e:
                    warnings.warn(f"Histogram matching failed for {map_type}: {e}")
                    matched_maps[map_type] = maps[map_type]
        
        return matched_maps
    
    @staticmethod
    def compute_correlation(map1: np.ndarray, map2: np.ndarray, 
                          method: str = 'pearson') -> float:
        """
        Compute correlation between two maps.
        
        Args:
            map1, map2: Input arrays
            method: 'pearson' or 'spearman'
        
        Returns:
            Correlation coefficient
        """
        # Flatten arrays and remove NaN values
        flat1 = map1.flatten()
        flat2 = map2.flatten()
        
        # Create mask for valid values
        valid_mask = ~(np.isnan(flat1) | np.isnan(flat2))
        
        if np.sum(valid_mask) < 10:  # Need minimum valid points
            return np.nan
        
        valid1 = flat1[valid_mask]
        valid2 = flat2[valid_mask]
        
        # Check for constant arrays
        if np.std(valid1) == 0 or np.std(valid2) == 0:
            return np.nan
        
        try:
            if method == 'pearson':
                corr, _ = pearsonr(valid1, valid2)
            elif method == 'spearman':
                corr, _ = spearmanr(valid1, valid2)
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            return corr if not np.isnan(corr) else 0.0
            
        except Exception as e:
            warnings.warn(f"Correlation computation failed: {e}")
            return np.nan
    
    def compute_pairwise_correlations(self, maps: Dict[str, np.ndarray],
                                    histogram_match: bool = False,
                                    histogram_reference: str = None,
                                    method: str = 'pearson',
                                    smoothing_sigma: Optional[float] = None,
                                    exclude_from_smoothing: Optional[List[str]] = None,
                                    exclude_from_matching: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute all pairwise correlations for a single image.

        Args:
            maps: Dictionary of map_type -> array
            histogram_match: Whether to apply histogram matching
            histogram_reference: Which map to use as reference for histogram matching
            method: Correlation method
            smoothing_sigma: If set, apply Gaussian smoothing (then range-renorm) to each
                map before correlating. sigma=15 replicates Henderson & Hayes (2025).
            exclude_from_smoothing: Map names to skip when smoothing (e.g. ['fixation']
                if fixation maps are already pre-smoothed density estimates).
            exclude_from_matching: Map names to leave unmatched during histogram matching
                (e.g. ['fixation'] to exclude fixations when matching to deepgaze).

        Returns:
            DataFrame with pairwise correlations
        """
        # Filter out original image from correlation computation
        correlation_maps = {k: v for k, v in maps.items() if k != '__image__'}

        if smoothing_sigma is not None:
            correlation_maps = {
                k: self.smooth_map(v, smoothing_sigma) if k not in (exclude_from_smoothing or []) else v
                for k, v in correlation_maps.items()
            }

        if histogram_match:
            correlation_maps = self.apply_histogram_matching(
                correlation_maps, histogram_reference, exclude_from_matching
            )

        map_types = list(correlation_maps.keys())
        n_types = len(map_types)
        
        # Initialize correlation matrix
        corr_matrix = np.full((n_types, n_types), np.nan)
        
        for i, type1 in enumerate(map_types):
            for j, type2 in enumerate(map_types):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:  # Compute only upper triangle
                    corr = self.compute_correlation(
                        correlation_maps[type1], 
                        correlation_maps[type2], 
                        method
                    )
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # Symmetric
        
        return pd.DataFrame(corr_matrix, index=map_types, columns=map_types)


class SaliencyVisualizer:
    """Handles all visualization functionality."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_image_comparison(self, image_name: str, maps: Dict[str, np.ndarray],
                            correlations: pd.DataFrame, save_path: str = None,
                            show_colorbar: bool = False):
        """
        Plot all maps for a single image with correlation matrix.
        
        Args:
            image_name: Name of the image
            maps: Dictionary of map_type -> array
            correlations: Correlation matrix
            save_path: Optional path to save figure
            show_colorbar: Whether to show colorbars
        """
        map_types = [k for k in maps.keys() if k != '__image__']
        n_maps = len(map_types)
        
        # Calculate layout
        n_cols = min(4, n_maps + 1)  # +1 for original
        n_rows = int(np.ceil((n_maps + 2) / n_cols))  # +2 for original and correlation matrix
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Plot original image
        if '__image__' in maps:
            axes_flat[0].imshow(maps['__image__'])
            axes_flat[0].set_title('Original Image', fontweight='bold')
            axes_flat[0].axis('off')
        
        # Plot saliency maps
        for i, map_type in enumerate(map_types):
            ax = axes_flat[i + 1]
            im = ax.imshow(maps[map_type], cmap='Reds', vmin=0, vmax=1)
            ax.set_title(f'{map_type.upper()}', fontweight='bold')
            ax.axis('off')
            
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot correlation matrix
        corr_ax = axes_flat[n_maps + 1]
        mask = np.triu(np.ones_like(correlations.values), k=1)
        sns.heatmap(
            correlations, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=corr_ax, mask=mask,
            cbar_kws={'shrink': 0.8},
            vmin=0, vmax=1
        )
        corr_ax.set_title('Pairwise Correlations', fontweight='bold')
        
        # Hide unused axes
        for i in range(n_maps + 2, len(axes_flat)):
            axes_flat[i].axis('off')
        
        plt.suptitle(f'Saliency Map Comparison - {image_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved image comparison to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, aggregated_correlations: pd.DataFrame,
                              save_path: str = None, title: str = None):
        """
        Plot aggregated correlation matrix across all images.
        
        Args:
            aggregated_correlations: Aggregated correlation matrix
            save_path: Optional path to save figure
            title: Optional custom title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(aggregated_correlations.values), k=1)
        
        # Plot heatmap
        sns.heatmap(aggregated_correlations, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, ax=ax, mask=mask,
                   cbar_kws={'label': 'Correlation Coefficient'}, 
                   vmin=0, vmax=1)
        
        if title is None:
            title = 'Aggregated Pairwise Correlations Across All Images'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")
        
        plt.show()


class SaliencyComparator:
    """Main interface class for saliency map comparison analysis."""

    def __init__(self, image_dir: str, map_configs: Dict[str, str]):
        """
        Initialize the saliency comparator.

        Args:
            image_dir: Directory containing original images
            map_configs: Dictionary mapping map type names to their directories
        """
        self.loader = SaliencyMapLoader(image_dir, map_configs)
        self.computer = CorrelationComputer()
        self.visualizer = SaliencyVisualizer()
        
        self.image_data = {}
        self.all_correlations = {}
        self.aggregated_correlations = None
        self.aggregation_method = 'median'  # Track which method was used
        
        print("SaliencyComparator initialized successfully")
        print(f"Image directory: {image_dir}")
        print(f"Map types: {list(map_configs.keys())}")

    def load_data(self):
        """Load all saliency map data."""
        self.image_data = self.loader.load_data()
        print(f"Data loaded for {len(self.image_data)} images")

    def compute_correlations(self, histogram_match: bool = False,
                           histogram_reference: str = None,
                           method: str = 'pearson',
                           aggregation_method: str = 'median',
                           smoothing_sigma: Optional[float] = None,
                           exclude_from_smoothing: Optional[List[str]] = None,
                           exclude_from_matching: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute correlations for all images.

        Args:
            histogram_match: Whether to apply histogram matching
            histogram_reference: Which map to use as reference for histogram matching
            method: Correlation method ('pearson' or 'spearman')
            aggregation_method: How to aggregate across images ('median' or 'mean')
            smoothing_sigma: If set, apply Gaussian smoothing (then range-renorm) to each
                map before correlating. sigma=15 replicates Henderson & Hayes (2025).
            exclude_from_smoothing: Map names to skip when smoothing (e.g. ['fixation']
                if fixation maps are already pre-smoothed density estimates).
            exclude_from_matching: Map names to leave unmatched during histogram matching
                (e.g. ['fixation'] to exclude fixations when matching to deepgaze).

        Returns:
            Dictionary of image_name -> correlation matrix
        """
        if not self.image_data:
            raise ValueError("No data loaded. Call load_data() first.")

        if aggregation_method not in ['median', 'mean']:
            raise ValueError("aggregation_method must be 'median' or 'mean'")

        print(f"Computing {method} correlations with histogram_match={histogram_match}, aggregation={aggregation_method}...")
        if histogram_match and histogram_reference:
            print(f"Using '{histogram_reference}' as histogram matching reference")
        if smoothing_sigma is not None:
            excluded_note = f", excluding {exclude_from_smoothing}" if exclude_from_smoothing else ""
            print(f"Applying Gaussian smoothing (sigma={smoothing_sigma}) before correlation{excluded_note}")
        if histogram_match and exclude_from_matching:
            print(f"Excluding {exclude_from_matching} from histogram matching")

        self.all_correlations = {}
        self.aggregation_method = aggregation_method
        correlation_matrices = []

        for img_name, maps in self.image_data.items():
            try:
                corr_matrix = self.computer.compute_pairwise_correlations(
                    maps, histogram_match, histogram_reference, method,
                    smoothing_sigma, exclude_from_smoothing, exclude_from_matching
                )
                self.all_correlations[img_name] = corr_matrix
                correlation_matrices.append(corr_matrix.values)
                
            except Exception as e:
                warnings.warn(f"Failed to compute correlations for {img_name}: {e}")
                continue

        # Compute aggregated correlations
        if correlation_matrices:
            correlation_stack = np.stack(correlation_matrices, axis=0)
            
            if aggregation_method == 'median':
                aggregated_values = np.median(correlation_stack, axis=0)
            else:  # mean
                aggregated_values = np.mean(correlation_stack, axis=0)
            
            # Get column/index names from first correlation matrix
            first_corr = next(iter(self.all_correlations.values()))
            self.aggregated_correlations = pd.DataFrame(
                aggregated_values, 
                index=first_corr.index, 
                columns=first_corr.columns
            )
            
            print(f"Correlations computed for {len(correlation_matrices)} images using {aggregation_method}")
        else:
            raise ValueError("No valid correlations computed")

        return self.all_correlations

    def get_mean_correlations(self) -> pd.DataFrame:
        """Get aggregated correlation matrix across all images."""
        if self.aggregated_correlations is None:
            raise ValueError("No correlations computed. Call compute_correlations() first.")
        return self.aggregated_correlations

    def get_aggregated_correlations(self) -> pd.DataFrame:
        """Get aggregated correlation matrix across all images."""
        return self.get_mean_correlations()

    def plot_image_comparison(self, image_name: str, save_path: str = None,
                            histogram_match: bool = False,
                            histogram_reference: str = None):
        """
        Plot comparison for a specific image.

        Args:
            image_name: Name of image to plot (without .png extension)
            save_path: Optional path to save figure
            histogram_match: Whether to apply histogram matching for display
            histogram_reference: Which map to use as reference for histogram matching
        """
        if image_name not in self.image_data:
            raise ValueError(f"Image '{image_name}' not found in loaded data")

        if image_name not in self.all_correlations:
            raise ValueError(f"No correlations computed for '{image_name}'. Call compute_correlations() first.")

        maps = self.image_data[image_name].copy()

        # Apply histogram matching if requested (for visualization only)
        if histogram_match:
            correlation_maps = {k: v for k, v in maps.items() if k != '__image__'}
            matched_maps = self.computer.apply_histogram_matching(correlation_maps, histogram_reference)
            maps.update(matched_maps)

        self.visualizer.plot_image_comparison(
            image_name, maps, self.all_correlations[image_name], save_path
        )

    def plot_correlation_matrix(self, save_path: str = None, title: str = None):
        """
        Plot aggregated correlation matrix across all images.

        Args:
            save_path: Optional path to save figure
            title: Optional custom title
        """
        if self.aggregated_correlations is None:
            raise ValueError("No aggregated correlations available. Call compute_correlations() first.")

        if title is None:
            title = f'{self.aggregation_method.capitalize()} Pairwise Correlations Across All Images'

        self.visualizer.plot_correlation_matrix(self.aggregated_correlations, save_path, title)

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics of correlations.

        Returns:
            Dictionary with summary statistics
        """
        if self.aggregated_correlations is None:
            raise ValueError("No correlations computed. Call compute_correlations() first.")

        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(self.aggregated_correlations.values), k=1)
        upper_triangle = self.aggregated_correlations.values[mask.astype(bool)]
        valid_correlations = upper_triangle[~np.isnan(upper_triangle)]

        if len(valid_correlations) == 0:
            return {"error": "No valid correlations found"}

        return {
            f"{self.aggregation_method}_correlation": np.median(valid_correlations) if self.aggregation_method == 'median' else np.mean(valid_correlations),
            "std_correlation": np.std(valid_correlations),
            "min_correlation": np.min(valid_correlations),
            "max_correlation": np.max(valid_correlations),
            "median_correlation": np.median(valid_correlations),
            "mean_correlation": np.mean(valid_correlations),
            "n_comparisons": len(valid_correlations),
            "aggregation_method": self.aggregation_method
        }

    def export_results(self, output_dir: str):
        """
        Export all results to files.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if self.aggregated_correlations is not None:
            # Save aggregated correlations
            filename = f"{self.aggregation_method}_correlations.csv"
            self.aggregated_correlations.to_csv(output_path / filename)
            
            # Save individual correlations
            for img_name, corr_matrix in self.all_correlations.items():
                corr_matrix.to_csv(output_path / f"correlations_{img_name}.csv")
            
            # Save summary statistics
            summary = self.get_summary_statistics()
            with open(output_path / "summary_statistics.txt", 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Results exported to {output_path}")


def add_statistical_comparison_to_comparator():
    """
    Extension method to add statistical comparison functionality to SaliencyComparator.
    Add this method to your SaliencyComparator class.
    """
    
    def get_statistical_comparator(self) -> StatisticalComparator:
        """Get statistical comparator instance."""
        if not self.all_correlations:
            raise ValueError("No correlations computed. Call compute_correlations() first.")
        return StatisticalComparator(self.all_correlations)

    def compare_with_reference(self, reference_map: str, test_maps: List[str] = None,
                              test_type: str = 'paired_t', alpha: float = 0.05,
                              plot: bool = True, save_path: str = None) -> pd.DataFrame:
        """
        Compare correlations of different maps against a reference map.

        Args:
            reference_map: Name of reference map (e.g., 'fixation')
            test_maps: List of maps to compare (if None, use all except reference)
            test_type: 'paired_t', 'wilcoxon', or 'mannwhitney'
            alpha: Significance level
            plot: Whether to create visualization
            save_path: Optional path to save plot

        Returns:
            DataFrame with comparison results
        """
        stat_comp = self.get_statistical_comparator()
        
        # Get all available maps if test_maps not specified
        if test_maps is None:
            all_maps = list(next(iter(self.all_correlations.values())).index)
            test_maps = [m for m in all_maps if m != reference_map]
        
        # Compute statistical comparisons
        results = stat_comp.compare_map_pairs(reference_map, test_maps, test_type, alpha)
        
        # Create visualization if requested
        if plot and not results.empty:
            stat_comp.plot_pairwise_comparison(
                reference_map, test_maps, results, test_type, save_path=save_path
            )
        
        # Print summary
        if not results.empty:
            summary_table = stat_comp.create_summary_table(reference_map, test_maps, test_type)
            print(f"\nStatistical Comparison Results ({test_type}):")
            print(f"Reference map: {reference_map}")
            print("=" * 80)
            print(summary_table.to_string(index=False))
            print("=" * 80)
            print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        else:
            print("No valid comparisons could be computed.")
        
        return results
    
    # Add methods to SaliencyComparator class
    SaliencyComparator.get_statistical_comparator = get_statistical_comparator
    SaliencyComparator.compare_with_reference = compare_with_reference


# Example usage and testing functions
def example_usage():
    """Example of how to use the SaliencyComparator."""
    
    # Configuration
    image_dir = "path/to/original/images"
    map_configs = {
        'aws': 'path/to/aws/maps',
        'gbvs': 'path/to/gbvs/maps', 
        'deepgaze': 'path/to/deepgaze/maps',
        'fixation': 'path/to/fixation/maps'
    }
    
    # Initialize comparator
    comparator = SaliencyComparator(image_dir, map_configs)
    
    # Load data
    comparator.load_data()
    
    # Compute correlations with histogram matching and median aggregation (default)
    # Using 'fixation' as the histogram matching reference
    correlations = comparator.compute_correlations(
        histogram_match=True, 
        histogram_reference='fixation',
        aggregation_method='median'
    )
    
    # Or use mean aggregation with 'gbvs' as histogram reference
    # correlations = comparator.compute_correlations(
    #     histogram_match=True, 
    #     histogram_reference='gbvs',
    #     aggregation_method='mean'
    # )
    
    # Get summary statistics
    stats = comparator.get_summary_statistics()
    print("Summary Statistics:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Plot correlation matrix
    comparator.plot_correlation_matrix(save_path="correlation_matrix.png")
    
    # Plot individual image comparison
    image_names = list(comparator.image_data.keys())
    if image_names:
        comparator.plot_image_comparison(
            image_names[0], 
            save_path="image_comparison.png",
            histogram_match=True,
            histogram_reference='fixation'
        )
    
    # Export all results
    comparator.export_results("output_results")


if __name__ == "__main__":
    print("Saliency Map Comparison Module")
    print("Import this module and use SaliencyComparator class")
    print("See example_usage() function for usage examples")