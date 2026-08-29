"""
Image Patch Extractor

This module extracts image patches from larger images based on coordinates 
provided in a CSV file. It supports both square and round patches.
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import glob
from pathlib import Path

def extract_patches(csv_file, image_source, save_dir, shape='round'):
    """
    Extract patches from images based on parameters in a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing patch parameters.
        image_source (str): Path to an image file or directory of images.
        save_dir (str): Directory where extracted patches will be saved.
        shape (str): Shape of the patch ('round' or 'square'). Default is 'round'.
    
    Returns:
        list: Paths to the extracted patch images.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Read patch parameters from CSV
    patch_params = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_params.append({
                'imw': int(row['imw']),
                'imh': int(row['imh']),
                'radius': int(row['radius']),
                'xc': int(row['xc']),
                'yc': int(row['yc'])
            })
    
    # Get list of image paths
    if os.path.isdir(image_source):
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(image_source, '**', ext), recursive=True))
    else:
        image_paths = [image_source]
    
    extracted_patches = []
    
    # Process each image
    for img_path in image_paths:
        img_filename = os.path.basename(img_path)
        img_name, img_ext = os.path.splitext(img_filename)
        
        try:
            # Open image and ensure it has an alpha channel
            img = Image.open(img_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Extract patches for this image using each set of parameters
            for params in patch_params:
                # Resize image if needed
                if img.width != params['imw'] or img.height != params['imh']:
                    img_resized = img.resize((params['imw'], params['imh']), Image.LANCZOS)
                else:
                    img_resized = img
                
                # Calculate patch dimensions - same as visualization
                imw = params['imw']
                imh = params['imh']
                radius = params['radius']
                xc = params['xc']
                yc = params['yc']
                
                # Create patch canvas with transparent background
                patch_size = radius * 2
                patch = Image.new('RGBA', (patch_size, patch_size), (0, 0, 0, 0))
                
                # Calculate source bounds in image coordinates (same as visualization)
                src_left = xc - radius
                src_top = yc - radius
                src_right = xc + radius
                src_bottom = yc + radius
                
                # Clip source bounds to actual image dimensions
                clipped_left = max(0, src_left)
                clipped_top = max(0, src_top)
                clipped_right = min(imw, src_right)
                clipped_bottom = min(imh, src_bottom)
                
                # Skip if no overlap with image
                if clipped_left >= clipped_right or clipped_top >= clipped_bottom:
                    print(f"Warning: Patch at ({xc}, {yc}) with radius {radius} has no overlap with the image.")
                    continue
                
                # Extract the overlapping region from the source image
                region = img_resized.crop((clipped_left, clipped_top, clipped_right, clipped_bottom))
                
                # Calculate where to place this region in the patch canvas
                # This accounts for cases where patch extends beyond image bounds
                dest_x = clipped_left - src_left  # Offset from left edge of patch
                dest_y = clipped_top - src_top    # Offset from top edge of patch
                
                # Paste the region onto the patch canvas
                patch.paste(region, (dest_x, dest_y))
                
                # Apply circular mask if shape is 'round'
                if shape.lower() == 'round':
                    # Create a circular mask
                    mask = Image.new('L', (patch_size, patch_size), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0, patch_size, patch_size), fill=255)
                    
                    # Create a blank transparent image for the final result
                    masked_patch = Image.new('RGBA', (patch_size, patch_size), (0, 0, 0, 0))
                    
                    # Apply the mask
                    masked_patch.paste(patch, (0, 0), mask)
                    patch = masked_patch
                
                # Save the patch
                patch_filename = f"{img_name}_w{imw}_h{imh}_x{xc}_y{yc}_r{radius}.png"
                patch_path = os.path.join(save_dir, patch_filename)
                patch.save(patch_path)
                extracted_patches.append(patch_path)
                
                print(f"Extracted patch from {img_filename}: center=({xc}, {yc}), radius={radius}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return extracted_patches

def visualize_patches(csv_file, image_path=None, shape='round'):
    """
    Visualize patch locations on an image or white canvas.
    
    Args:
        csv_file (str): Path to the CSV file containing patch parameters.
        image_path (str, optional): Path to the image. If None, creates a white canvas
                                   with dimensions from the CSV file.
        shape (str): Shape of the patch ('round' or 'square'). Default is 'round'.
    """
    # Read patch parameters from CSV
    patch_params = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_params.append({
                'imw': int(row['imw']),
                'imh': int(row['imh']),
                'radius': int(row['radius']),
                'xc': int(row['xc']),
                'yc': int(row['yc'])
            })
    
    if not patch_params:
        raise ValueError("No patch parameters found in the CSV file")
    
    # Get dimensions from the first entry in the CSV
    img_width = patch_params[0]['imw']
    img_height = patch_params[0]['imh']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if image_path is None:
        # Create a white canvas with dimensions from CSV
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Invert y-axis to match image coordinates
        ax.set_facecolor('white')
        ax.set_title('Patch Visualization on White Canvas')
    else:
        # Check if image_path is a directory
        if os.path.isdir(image_path):
            # Find the first image in the directory
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(os.path.join(image_path, ext)))
            
            if not image_files:
                print(f"No image files found in directory {image_path}, using white canvas instead")
                ax.set_xlim(0, img_width)
                ax.set_ylim(img_height, 0)
                ax.set_facecolor('white')
                ax.set_title('Patch Visualization on White Canvas (No images found)')
            else:
                image_path = image_files[0]
                # Open and display the image
                img = Image.open(image_path)
                
                # Check if resize is needed
                if img.width != img_width or img.height != img_height:
                    img = img.resize((img_width, img_height), Image.LANCZOS)
                
                ax.imshow(img)
                ax.set_title(f'Patch Visualization - {os.path.basename(image_path)}')
        else:
            # Open and display the image
            img = Image.open(image_path)
            
            # Check if resize is needed
            if img.width != img_width or img.height != img_height:
                img = img.resize((img_width, img_height), Image.LANCZOS)
            
            ax.imshow(img)
            ax.set_title(f'Patch Visualization - {os.path.basename(image_path)}')
    
    # Get unique radius values for color coding
    unique_radii = sorted(set(p['radius'] for p in patch_params))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_radii)))
    radius_to_color = {r: c for r, c in zip(unique_radii, colors)}
    
    # Plot each patch
    for params in patch_params:
        radius = params['radius']
        xc = params['xc']
        yc = params['yc']
        color = radius_to_color[radius]
        
        if shape.lower() == 'round':
            # Draw circle
            circle = Circle((xc, yc), radius, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(circle)
        else:
            # Draw square
            rect = Rectangle((xc - radius, yc - radius), 2*radius, 2*radius, 
                             fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
        
        # Draw center
        ax.plot(xc, yc, 'o', color=color, markersize=4)
    
    # Create a legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=radius_to_color[r], 
                              markersize=10, label=f'Radius {r}') for r in unique_radii]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()