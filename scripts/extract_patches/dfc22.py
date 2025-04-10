# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import rasterio
from itertools import product
from rasterio.windows import Window
import numpy as np
import pandas as pd


def extract_patches(indices_file, source_dir, output_dir, patch_size, test):
    """Extract patches from DFC2022 dataset source files.
    
    This function reads patch coordinates from an indices file, then extracts
    corresponding patches from the source imagery (BDORTHO, RGEALTI, UrbanAtlas)
    and saves them as GeoTIFF files.
    
    Args:
        indices_file: Path to CSV file with patch indices
        source_dir: Directory containing source imagery
        output_dir: Directory to save extracted patches
        patch_size: Size of patches to extract (width and height)
        test: If True, process only a small subset of indices
        
    Returns:
        List of missing files that couldn't be processed
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    indices = pd.read_csv(indices_file)

    if test:
        indices = indices.sample(100)

    missing = list()

    for _, row in tqdm(indices.iterrows()):
        # Extract patch information from the row
        city, img_name, split, xmin, ymin = (
            row["city"],
            row["tile"],
            row["split"],
            int(row["xmin"]),
            int(row["ymin"]),
        )

        # Create paths to source files
        bdortho_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "BDORTHO"
            / f"{img_name}.tif"
        )
        rgealti_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "RGEALTI"
            / f"{img_name}_RGEALTI.tif"
        )
        urbanatlas_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "UrbanAtlas"
            / f"{img_name}_UA2012.tif"
        )

        # Skip if any source file is missing
        if (
            not bdortho_path.exists()
            or not rgealti_path.exists()
            or not urbanatlas_path.exists()
        ):
            missing.append((split, city, img_name))
            continue

        with rasterio.open(bdortho_path) as bdortho_src, rasterio.open(
            rgealti_path
        ) as rgealti_src, rasterio.open(urbanatlas_path) as urbanatlas_src:
            # Resample DEM to match RGB resolution
            dem_resampled = rgealti_src.read(
                out_shape=(bdortho_src.height, bdortho_src.width),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0]

            if (
                xmin == -1 and ymin == -1
            ):  # Special case: extract all patches with a stride
                stride = 200
                for i, j in product(
                    range(0, bdortho_src.width - stride, stride),
                    range(0, bdortho_src.height - stride, stride),
                ):
                    # Create a window for this patch
                    window = Window.from_slices(
                        (j, j + patch_size), (i, i + patch_size)
                    )
                    # Extract RGB, DEM, and mask data
                    rgb = bdortho_src.read(window=window)
                    dem_patch = dem_resampled[j : j + patch_size, i : i + patch_size]
                    mask = urbanatlas_src.read(window=window)
                    # Stack all bands together (RGB + DEM + mask)
                    patch = np.vstack([rgb, dem_patch[np.newaxis, :, :], mask])
                    out_file_path = (
                        output_dir / split / f"{city}_{img_name}_{i}_{j}.tif"
                    )
                    # Preserve georeferencing information
                    transform = bdortho_src.window_transform(window)
                    meta = bdortho_src.meta.copy()
                    meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "count": patch.shape[0],
                            "dtype": patch.dtype,
                            "transform": transform,
                        }
                    )
                    out_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_file_path, "w", **meta) as dst:
                        dst.write(patch)
            else:  # Extract a single patch at the specified coordinates
                window = Window.from_slices(
                    (xmin, xmin + patch_size), (ymin, ymin + patch_size)
                )
                rgb = bdortho_src.read(window=window)
                dem_patch = dem_resampled[
                    xmin : xmin + patch_size, ymin : ymin + patch_size
                ]
                mask = urbanatlas_src.read(window=window)
                patch = np.vstack([rgb, dem_patch[np.newaxis, :, :], mask])
                out_file_path = (
                    output_dir / split / f"{city}_{img_name}_{xmin}_{ymin}.tif"
                )
                transform = bdortho_src.window_transform(window)
                meta = bdortho_src.meta.copy()
                meta.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "count": patch.shape[0],
                        "dtype": patch.dtype,
                        "transform": transform,
                    }
                )
                out_file_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_file_path, "w", **meta) as dst:
                    dst.write(patch)
    return missing


def main():
    """Parse command-line arguments and run patch extraction."""
    parser = argparse.ArgumentParser(description="Extract geospatial patches.")
    parser.add_argument(
        "--indices_file", type=str, required=True, help="File containing patch indices."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing the original TIF files to be cropped.",
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
    missing = extract_patches(
        args.indices_file, args.source_dir, args.output_dir, args.patch_size, args.test
    )

    # Write list of missing files to a text file
    with open(os.path.join(args.output_dir, "missing.txt"), "w") as f:
        for split, city, img_name in missing:
            f.write(f"{split}, {city}, {img_name}\n")


if __name__ == "__main__":
    main()
