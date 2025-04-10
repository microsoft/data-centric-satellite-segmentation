# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
from concurrent.futures import ProcessPoolExecutor
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_samples


def process_tile_censored(data, root_str):
    """Determine if a tile is censored based on mask content.
    
    Classifies a tile as censored (0) if more pixels belong to class 0 (no data)
    than to other classes.
    
    Args:
        data: Dictionary containing tile information (id, xmin, ymin)
        root_str: Root directory containing the tiles
        
    Returns:
        0 if tile is censored (majority no data), 1 otherwise
        None if there was an error processing the tile
    """
    root_path = Path(root_str)
    tile_path = (
        root_path / f"{data['id']}_{data['xmin']}_{data['ymin']}.tif"
    )
    try:
        with rasterio.open(tile_path) as src:
            mask = src.read(5)  # Read mask band
        # Remap class 15 (clouds/shadows) to class 0 (no data)
        mask = np.where(mask == 15, 0, mask)
        # Return 0 if more pixels are no data than data, 1 otherwise
        return 0 if np.count_nonzero(mask == 0) > np.count_nonzero(mask != 0) else 1
    except Exception as e:
        print(f"Error processing {tile_path}: {e}")
        return None


def process_tile_balanced(data, root_str):
    """Process a tile for balanced class distribution sampling.
    
    Identifies the most common class in a tile and rates tile quality based on
    class distribution.
    
    Args:
        data: Dictionary containing tile information (id, xmin, ymin)
        root_str: Root directory containing the tiles
        
    Returns:
        Most common class index, or second most common if most common is 0 and over 50%
    """
    root_path = Path(root_str)
    tile, xmin, ymin = data["id"], data["xmin"], data["ymin"]
    tile_path = root_path / f"{tile}_{xmin}_{ymin}.tif"
    with rasterio.open(tile_path) as src:
        mask = src.read(5).astype("uint8").flatten()
        # Remap classes 8, 9, 15 to class 0
        mask = np.where(np.isin(mask, [8, 9, 15]), 0, mask)
    # Find most frequent class
    cls_val = np.bincount(mask).argmax()
    # Calculate percentage of pixels belonging to most frequent class
    ratio = np.count_nonzero(mask == cls_val) / len(mask)
    # If most frequent class is 0 and ratio > 0.5, return second most frequent class
    if cls_val == 0:
        return 0 if ratio > 0.5 else np.bincount(mask).argsort()[-2]
    else:
        return cls_val


def random_mode(output_path, root_dir):
    """Generate a CSV file with random scoring.
    
    Args:
        output_path: Path to save the output CSV file
        root_dir: Root directory containing the tiles
    """
    samples = get_samples(root_dir=root_dir)
    # Randomly shuffle samples
    samples = samples.sample(frac=1)
    # Assign scores linearly from 1 to 0
    samples["score"] = np.linspace(1, 0, len(samples))
    samples.to_csv(output_path)


def censored_mode(output_path, root_dir):
    """Generate a CSV file with censored tiles ranked lowest.
    
    Identifies censored tiles (those with majority no data) and ranks them
    below non-censored tiles.
    
    Args:
        output_path: Path to save the output CSV file
        root_dir: Root directory containing the tiles
    """
    d = get_samples(root_dir=root_dir)
    data_dicts = d.to_dict("records")
    total = len(data_dicts)
    
    pbar = tqdm(total=total, desc="Processing tiles")
    
    # Process all tiles in parallel to determine if they are censored
    scores = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tile_censored, data, str(root_dir)) for data in data_dicts]
        for future in futures:
            scores.append(future.result())
            pbar.update(1)
    
    pbar.close()
    
    if any(score is None for score in scores):
        raise ValueError("Some tiles resulted in errors. Check the logs.")
    
    d["score"] = scores
    # For non-censored tiles (score=1), assign random scores between 0 and 1
    # This ensures they're all ranked higher than censored tiles (score=0)
    d.loc[d["score"] == 1, "score"] = np.random.uniform(0, 1, len(d[d["score"] == 1]))
    # Sort by score in descending order
    d = d.sort_values("score", ascending=False)
    # Rescale scores to range from 1 to 0
    d["score"] = np.linspace(1, 0, len(d))
    d.to_csv(output_path)


def balanced_mode(output_path, root_dir):
    """Generate a CSV file with balanced class distribution.
    
    Groups tiles by their majority class and ensures a balanced representation of classes
    in the ranking, prioritizing non-censored data.
    
    Args:
        output_path: Path to save the output CSV file
        root_dir: Root directory containing the tiles
    """
    d = get_samples(root_dir=root_dir)
    data_dicts = d.to_dict("records")
    total = len(data_dicts)
    pbar = tqdm(total=total, desc="Processing tiles")
    
    # Process all tiles in parallel to identify class distributions
    groups = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_tile_balanced, data, str(root_dir)) for data in data_dicts]
        for future in futures:
            groups.append(future.result())
            pbar.update(1)
    
    pbar.close()
    
    d["group"] = groups
    # Separate censored tiles (group=0) from others
    black_listed = d[d["group"] == 0]
    group_samples = d[d["group"] != 0]
    
    # Add counting column to ensure balanced sampling from each group
    group_samples["cumcount"] = group_samples.groupby("group").cumcount()
    
    # Sort first by cumcount (round-robin through groups) then by group number
    rank = group_samples.sort_values(by=["cumcount", "group"]).drop(
        ["cumcount", "group"], axis=1
    )
    # Add censored tiles at the end
    rank = pd.concat([rank, black_listed.drop("group", axis=1)], axis=0)
    # Rescale scores to range from 1 to 0
    rank["score"] = np.linspace(1, 0, len(rank))
    rank.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score and export CSV files of samples."
    )
    parser.add_argument(
        "--root_dir", required=True, help="Path to the directory containing the tile files."
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["random", "censored", "balanced"],
        help="Scoring mode.",
    )

    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_path).parent
    if not output_dir.exists(): output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "random":
        random_mode(args.output_path, root_dir)
    elif args.mode == "censored":
        censored_mode(args.output_path, root_dir)
    elif args.mode == "balanced":
        balanced_mode(args.output_path, root_dir)
    else:
        raise ValueError(
            "Invalid mode. Choose from 'random', 'censored', or 'balanced'."
        )
