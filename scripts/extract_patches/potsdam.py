# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
from itertools import product
from rasterio.windows import Window
import numpy as np
import pandas as pd


def encode_mask(mask):
    """Convert RGB mask representation to class indices.
    
    The Potsdam dataset uses an RGB encoding for its masks, where different 
    color combinations represent different semantic classes. This function 
    converts the RGB representation to single-channel class indices.
    
    Args:
        mask: RGB mask array with shape (3, height, width)
        
    Returns:
        Single-channel class index array with shape (1, height, width)
    """
    mask = np.rollaxis(mask, 0, 3)  # Convert to (height, width, 3)
    new = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
    mask = mask // 255  # Normalize to binary (0 or 1) per channel
    
    # Use weighted sums to create unique values for each color combination
    # R = 1, G = 7, B = 49
    mask = mask * (1, 7, 49)
    mask = mask.sum(axis=2)
    
    # Map weighted sums to class indices
    new[mask == 1 + 7 + 49] = 0
    new[mask == 49] = 1
    new[mask == 7 + 49] = 2
    new[mask == 7] = 3
    new[mask == 1 + 7] = 4
    new[mask == 1] = 5
    
    return new[np.newaxis, :, :]  # Add channel dimension


def extract_patches(
    indices_file, rgb_dir, dsm_dir, masks_dir, output_dir, patch_size, test
):
    """Extract patches from Potsdam dataset source files.
    
    This function reads patch coordinates from an indices file and extracts
    corresponding patches from the RGB, DSM, and mask files, saving them
    as combined GeoTIFF files.
    
    Args:
        indices_file: Path to file containing patch indices
        rgb_dir: Directory containing RGB images
        dsm_dir: Directory containing DSM (height) images
        masks_dir: Directory containing ground truth masks
        output_dir: Directory to save extracted patches
        patch_size: Size of patches to extract (width and height)
        test: If True, process only a small subset of indices
    """
    rgb_dir = Path(rgb_dir)
    dsm_dir = Path(dsm_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load indices file with columns: img, dsm, mask, xmin, ymin, score
    indices = pd.read_csv(
        indices_file,
        sep=" ",
        header=None,
        names=["img", "dsm", "mask", "xmin", "ymin", "score"],
    )

    if test:
        indices = indices.head(100)
    for _, row in tqdm(indices.iterrows(), total=len(indices)):
        # Create file paths for RGB, DSM, and mask files
        img_fp = rgb_dir / f"{row['img']}.tif"
        dsm_fp = dsm_dir / f"{row['dsm']}.jpg"
        mask_fp = masks_dir / f"{row['mask']}.tif"

        assert img_fp.exists(), f"File does not exist: {img_fp}"
        assert dsm_fp.exists(), f"File does not exist: {dsm_fp}"
        assert mask_fp.exists(), f"File does not exist: {mask_fp}"

        # Extract tile ID from filename
        img_els = img_fp.stem.split("_")
        id0, id1 = img_els[2], img_els[3]
        img_id = f"{id0}_{id1}"

        xmin, ymin = int(row["xmin"]), int(row["ymin"])

        with rasterio.open(img_fp) as img_src, rasterio.open(
            dsm_fp
        ) as dsm_src, rasterio.open(mask_fp) as mask_src:
            # Resample DSM to match RGB resolution
            dsm_resampled = dsm_src.read(
                out_shape=(img_src.height, img_src.width),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0]

            if xmin == -1 and ymin == -1:  # Special case: extract all patches with a stride
                stride = 200
                for i, j in product(
                    range(0, img_src.width - stride, stride),
                    range(0, img_src.height - stride, stride),
                ):
                    # Define the window for this patch
                    window = Window.from_slices(
                        (j, j + patch_size), (i, i + patch_size)
                    )
                    # Extract RGB, DSM, and mask data
                    rgb = img_src.read(window=window)
                    dsm_patch = dsm_resampled[j : j + patch_size, i : i + patch_size]
                    mask = encode_mask(mask_src.read(window=window))
                    # Stack all bands (RGB + DSM + mask)
                    patch = np.vstack([rgb, dsm_patch[np.newaxis, :, :], mask])
                    out_file_path = output_dir / f"{img_id}_{i}_{j}.tif"
                    # Preserve georeferencing
                    transform = img_src.window_transform(window)
                    meta = img_src.meta.copy()
                    meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "count": patch.shape[0],
                            "dtype": patch.dtype,
                            "transform": transform,
                        }
                    )
                    with rasterio.open(out_file_path, "w", **meta) as dst:
                        dst.write(patch)
            else:  # Extract a single patch at the specified coordinates
                window = Window.from_slices(
                    (xmin, xmin + patch_size), (ymin, ymin + patch_size)
                )
                rgb = img_src.read(window=window)
                dsm_patch = dsm_resampled[
                    xmin : xmin + patch_size, ymin : ymin + patch_size
                ]
                mask = encode_mask(mask_src.read(window=window))
                patch = np.vstack([rgb, dsm_patch[np.newaxis, :, :], mask])
                out_file_path = output_dir / f"{img_id}_{xmin}_{ymin}.tif"
                transform = img_src.window_transform(window)
                meta = img_src.meta.copy()
                meta.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "count": patch.shape[0],
                        "dtype": patch.dtype,
                        "transform": transform,
                    }
                )
                with rasterio.open(out_file_path, "w", **meta) as dst:
                    dst.write(patch)


def main():
    """Parse command-line arguments and run patch extraction."""
    parser = argparse.ArgumentParser(description="Extract geospatial patches.")
    parser.add_argument(
        "--indices_file", type=str, required=True, help="File containing patch indices."
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        help="Directory containing the RGB TIF files to be cropped.",
    )
    parser.add_argument(
        "--dsm_dir",
        type=str,
        help="Directory containing the DSM TIF files to be cropped.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        help="Directory containing the masks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where patches will be saved.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Size of each square patch."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the script in test mode."
    )

    args = parser.parse_args()
    extract_patches(
        args.indices_file,
        args.rgb_dir,
        args.dsm_dir,
        args.masks_dir,
        args.output_dir,
        args.patch_size,
        args.test,
    )


if __name__ == "__main__":
    main()
