# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import entropy, get_samples

def get_lbp_representation(mask):
    """Create Local Binary Pattern (LBP) representation of a mask.
    
    LBP compares each pixel with its 8 neighbors and creates a binary code
    based on whether neighboring pixels have the same class as the center pixel.
    
    Args:
        mask: Input mask array
        
    Returns:
        Array of LBP codes for each pixel
    """
    height, width = mask.shape
    offsets = []
    # Iterate through all 8 neighboring directions
    for y_offset in [-1, 0, 1]:
        for x_offset in [-1, 0, 1]:
            if y_offset == 0 and x_offset == 0:
                continue
            # Compare center pixels with neighbor pixels
            comparisons = (
                mask[1 : (height - 1), 1 : (width - 1)]
                == mask[
                    1 + y_offset : (height - 1) + y_offset,
                    1 + x_offset : (width - 1) + x_offset,
                ]
            )
            offsets.append(comparisons)
    offsets = np.array(offsets)

    # Convert 8 binary comparisons to an 8-bit code
    binary_representation = np.zeros((height - 2, width - 2), dtype=np.uint8)
    for i in range(8):
        binary_representation |= offsets[i].astype(np.uint8) << i
    return binary_representation


def lbp_representation_entropy(data, root_path):
    """Calculate entropy of the LBP representation for a mask.
    
    Higher entropy indicates more complex boundaries between classes.
    
    Args:
        data: Dictionary containing tile information
        root_path: Root directory containing the tiles
        
    Returns:
        Entropy value for the LBP representation
    """
    tile, xmin, ymin = data["id"], data["xmin"], data["ymin"]
    tile_path = root_path / f"{tile}_{xmin}_{ymin}.tif"
    with rasterio.open(tile_path) as src:
        mask = src.read(5).astype("uint8")
    binary_representation = get_lbp_representation(mask)
    # Calculate histogram of LBP values
    counts = np.bincount(binary_representation.ravel(), minlength=256)
    counts = counts / counts.sum()
    return entropy(counts)


def process_tile_censored(data, root_str):
    """Determine if a tile is censored based on mask content.
    
    Classifies a tile as censored (0) if more pixels belong to class 0 (no data)
    than to other classes.
    
    Args:
        data: Dictionary containing tile information
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
            mask = src.read(5)
        mask = np.where(mask == 15, 0, mask)
        return 0 if np.count_nonzero(mask == 0) > np.count_nonzero(mask != 0) else 1
    except Exception as e:
        print(f"Error processing {tile_path}: {e}")
        return None


def calculate_entropy(store, data, num_classes=16):
    """Calculate entropy of class distribution in a patch.
    
    Higher entropy indicates more class diversity.
    
    Args:
        store: Path to the directory containing tiles
        data: Dictionary containing tile information
        num_classes: Number of possible classes
        
    Returns:
        Entropy value for the class distribution
    """
    img_id, xmin, ymin = data["id"], data["xmin"], data["ymin"]
    tile_path = store / f"{img_id}_{xmin}_{ymin}.tif"
    with rasterio.open(tile_path) as src:
        mask = src.read(src.count).astype("uint8").flatten()
    # Count occurrences of each class
    counts = np.bincount(mask, minlength=num_classes)
    if num_classes == 16:
        # Ignore class 0 (no data) and class 15 (clouds/shadows)
        counts[0], counts[num_classes - 1] = 0, 0
    total_count = counts.sum()
    if total_count == 0:
        return 0
    else:
        # Calculate entropy of the class distribution
        probs = counts / total_count
        return -np.sum(probs * np.log2(probs + 1e-9))


def get_image(file_path):
    """Load an image from a file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        RGB image array normalized to [0,1]
    """
    try:
        with rasterio.open(file_path) as f:
            array = f.read()
        if array.shape[0] >= 3:
            rgb_array = array[:3, :, :]
            return np.rollaxis(rgb_array, 0, 3) / 255.0
        else:
            # If less than 3 channels, duplicate the first channel
            first_band = array[0:1, :, :]
            rgb_array = np.concatenate([first_band, first_band, first_band], axis=0)
            return np.rollaxis(rgb_array, 0, 3) / 255.0
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return np.ones((10, 10, 3)) * np.array([1, 0, 0])  # Return red square on error


def plot_top_patches(df, store_path, top_n=100, png="top_patches.png"):
    """Plot the top-ranked patches in a grid.
    
    Args:
        df: DataFrame containing ranked patches
        store_path: Path to the directory containing tiles
        top_n: Number of top patches to display
        png: Output filename for the visualization
    """
    top_patches = df.head(top_n)
    grid_size = int(np.ceil(np.sqrt(top_n)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axs = axs.flatten()
    for i, (_, row) in enumerate(top_patches.iterrows()):
        if i < top_n:
            img_id, xmin, ymin = row["id"], row["xmin"], row["ymin"]
            file_path = store_path / f"{img_id}_{xmin}_{ymin}.tif"
            try:
                img = get_image(file_path)
                axs[i].imshow(img)
                axs[i].axis('off')
                axs[i].set_title(f"Rank: {i+1}", fontsize=8)
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
                axs[i].text(0.5, 0.5, f"Error loading image", ha='center', va='center')
                axs[i].axis('off')
    
    # Hide empty subplots
    for i in range(top_n, grid_size * grid_size):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(png, dpi=100, bbox_inches='tight', pad_inches=0.1, format='png')
    print(f"Saved top {top_n} patches visualization to {png}")


def mask_entropy_mode(output_path, dataset, root_path, png=None):
    """Generate rankings based on mask entropy (class diversity).
    
    Tiles with higher class diversity will be ranked higher.
    
    Args:
        output_path: Path to save the output CSV file
        dataset: Dataset name ('dfc', 'potsdam', or 'vaihingen')
        root_path: Root directory containing the datasets
        png: Optional path to save visualization of top patches
    """
    store = root_path / dataset / "train"
    d = get_samples(store)
    data_dicts = d.to_dict("records")
    if dataset == "dfc":
        print("Processing censored tiles...")
        with ProcessPoolExecutor() as executor:
            scores = list(
                executor.map(
                    process_tile_censored, data_dicts, [str(store)] * len(data_dicts)
                )
            )
        if any(score is None for score in scores):
            raise ValueError("Some tiles resulted in errors. Check the logs.")
        d["score"] = scores
        d["score"] = d["score"].replace({0: -1})  # Mark censored tiles with -1
    banned = d[d["score"] == -1].copy()  # Separate censored tiles
    allowed = d[d["score"] == 1].copy()  # Keep only non-censored tiles
    data_dicts = allowed.to_dict("records")
    if dataset == "dfc":
        num_classes = 16
    elif dataset == "potsdam":
        num_classes = 6
    else:
        num_classes = 6
    
    print(f"Calculating entropy for {len(data_dicts)} tiles...")
    with tqdm(total=len(data_dicts), desc="Calculating entropy") as pbar:
        with ProcessPoolExecutor() as executor:
            scores = []
            for score in executor.map(
                calculate_entropy,
                [store] * len(data_dicts),
                data_dicts,
                [num_classes] * len(data_dicts),
            ):
                scores.append(score)
                pbar.update(1)
    
    allowed["score"] = scores
    # Prevent zero weights by setting a minimum score
    allowed.loc[allowed["score"] <= 0, "score"] = 1e-9
    # Convert scores to weights for weighted sampling
    allowed["weight"] = allowed["score"] / allowed["score"].sum()
    rank = pd.DataFrame(columns=allowed.columns)
    
    # Perform weighted sampling without replacement based on entropy
    print("Ranking tiles by entropy...")
    with tqdm(total=len(allowed), desc="Ranking tiles") as pbar:
        while allowed.shape[0] > 0:
            sample = allowed.sample(1, weights="weight")
            rank = pd.concat([rank, sample])
            allowed = allowed.drop(sample.index)
            if len(allowed) > 0:
                allowed["weight"] = allowed["weight"] / allowed["weight"].sum()
            pbar.update(1)
    
    rank = rank.drop("weight", axis=1)
    # Add censored tiles at the end
    rank = pd.concat([rank, banned])
    # Assign final scores from 1 to 0
    rank["score"] = np.linspace(1, 0, len(rank))
    rank.to_csv(output_path)
    if png:
        plot_top_patches(rank, store_path=store, png=png)


def mask_lbp_mode(output_path, dataset, root_path, png=None):
    """Generate rankings based on LBP entropy (boundary complexity).
    
    Tiles with more complex boundaries between classes will be ranked higher.
    
    Args:
        output_path: Path to save the output CSV file
        dataset: Dataset name ('dfc', 'potsdam', or 'vaihingen')
        root_path: Root directory containing the datasets
        png: Optional path to save visualization of top patches
    """
    store = root_path / dataset / "train"
    d = get_samples(store)
    data_dicts = d.to_dict("records")
    print(f"Processing censored tiles...")
    with ProcessPoolExecutor() as executor:
        scores = list(
            executor.map(
                process_tile_censored, data_dicts, [str(store)] * len(data_dicts)
            )
        )
    if any(score is None for score in scores):
        raise ValueError("Some tiles resulted in errors. Check the logs.")
    d["score"] = scores
    d["score"] = d["score"].replace({0: -1})  # Mark censored tiles with -1
    banned = d[d["score"] == -1].copy()  # Separate censored tiles
    allowed = d[d["score"] == 1].copy()  # Keep only non-censored tiles
    data_dicts = allowed.to_dict("records")
    
    print(f"Calculating LBP representation for {len(data_dicts)} tiles...")
    with tqdm(total=len(data_dicts), desc="Calculating LBP") as pbar:
        with ProcessPoolExecutor() as executor:
            scores = []
            for score in executor.map(
                lbp_representation_entropy, 
                data_dicts, 
                [store] * len(data_dicts)
            ):
                scores.append(score)
                pbar.update(1)
    
    allowed["score"] = scores
    # Prevent zero weights by setting a minimum score
    allowed.loc[allowed["score"] <= 0, "score"] = 1e-9
    # Convert scores to weights for weighted sampling
    allowed["weight"] = allowed["score"] / allowed["score"].sum()
    rank = pd.DataFrame(columns=allowed.columns)
    
    # Perform weighted sampling without replacement based on LBP entropy
    print("Ranking tiles by LBP score...")
    with tqdm(total=len(allowed), desc="Ranking tiles") as pbar:
        while allowed.shape[0] > 0:
            sample = allowed.sample(1, weights="weight")
            rank = pd.concat([rank, sample])
            allowed = allowed.drop(sample.index)
            if len(allowed) > 0:
                allowed["weight"] = allowed["weight"] / allowed["weight"].sum()
            pbar.update(1)
    
    rank = rank.drop("weight", axis=1)
    # Add censored tiles at the end
    rank = pd.concat([rank, banned])
    # Assign final scores from 1 to 0
    rank["score"] = np.linspace(1, 0, len(rank))
    rank.to_csv(output_path)
    
    if png:
        plot_top_patches(rank, store_path=store, png=png)


def mask_hybrid_mode(output_path, dataset, root_path, png=None):
    """Generate rankings using a hybrid of entropy and LBP methods.
    
    Combines class diversity and boundary complexity metrics for ranking.
    
    Args:
        output_path: Path to save the output CSV file
        dataset: Dataset name ('dfc', 'potsdam', or 'vaihingen')
        root_path: Root directory containing the datasets
        png: Optional path to save visualization of top patches
    """
    store = root_path / dataset / "train"
    d = get_samples(store)
    data_dicts = d.to_dict("records")
    print(f"Processing censored tiles...")
    with ProcessPoolExecutor() as executor:
        scores = list(
            executor.map(
                process_tile_censored, data_dicts, [str(store)] * len(data_dicts)
            )
        )
    if any(score is None for score in scores):
        raise ValueError("Some tiles resulted in errors. Check the logs.")
    d["score"] = scores
    d["score"] = d["score"].replace({0: -1})  # Mark censored tiles with -1
    banned = d[d["score"] == -1].copy()  # Separate censored tiles
    allowed = d[d["score"] == 1].copy()  # Keep only non-censored tiles
    data_dicts = allowed.to_dict("records")
    
    print(f"Calculating entropy for {len(data_dicts)} tiles...")
    with tqdm(total=len(data_dicts), desc="Calculating entropy") as pbar:
        with ProcessPoolExecutor() as executor:
            entropies = []
            for entropy in executor.map(calculate_entropy, [store] * len(data_dicts), data_dicts):
                entropies.append(entropy)
                pbar.update(1)
    
    print(f"Calculating LBP representation for {len(data_dicts)} tiles...")
    with tqdm(total=len(data_dicts), desc="Calculating LBP") as pbar:
        with ProcessPoolExecutor() as executor:
            lbps = []
            for lbp in executor.map(
                lbp_representation_entropy, 
                data_dicts,
                [store] * len(data_dicts)
            ):
                lbps.append(lbp)
                pbar.update(1)
    
    allowed["lbp"] = lbps
    allowed["entropy"] = entropies
    # Normalize both scores to [0,1] range
    allowed["lbp"] /= allowed["lbp"].max()
    allowed["entropy"] /= allowed["entropy"].max()
    # Combine the two metrics (with small epsilon to avoid zeros)
    allowed["weight"] = (allowed["entropy"] + 1e-4) * (allowed["lbp"] + 1e-4)
    allowed["weight"] /= allowed["weight"].sum()
    allowed = allowed.drop(["entropy", "lbp"], axis=1)
    rank = pd.DataFrame(columns=allowed.columns)
    
    # Perform weighted sampling without replacement based on hybrid score
    print("Ranking tiles by hybrid score...")
    with tqdm(total=len(allowed), desc="Ranking tiles") as pbar:
        while allowed.shape[0] > 0:
            sample = allowed.sample(1, weights="weight")
            rank = pd.concat([rank, sample])
            allowed = allowed.drop(sample.index)
            if len(allowed) > 0:
                allowed["weight"] /= allowed["weight"].sum()
            pbar.update(1)
    
    rank = rank.drop("weight", axis=1)
    # Add censored tiles at the end
    rank = pd.concat([rank, banned])
    # Assign final scores from 1 to 0
    rank["score"] = np.linspace(1, 0, len(rank))
    rank.to_csv(output_path)
    
    if png:
        plot_top_patches(rank, store_path=store, png=png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score and export CSV files of samples."
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save the output CSV file."
    )
    parser.add_argument("--dataset", required=True, help="Which dataset to use.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["entropy", "lbp", "hybrid"],
        help="Scoring mode.",
    )
    parser.add_argument(
        "--root_dir", 
        required=True, 
        help="Root directory containing the image datasets."
    )
    parser.add_argument(
        "--png",
        required=False,
        help="Path to save the PNG file with the top 100 ranked patches.",
    )

    args = parser.parse_args()
    root_path = Path(args.root_dir)

    if args.mode == "entropy":
        mask_entropy_mode(args.output_path, args.dataset, root_path, args.png)
    elif args.mode == "lbp":
        mask_lbp_mode(args.output_path, args.dataset, root_path, args.png)
    elif args.mode == "hybrid":
        mask_hybrid_mode(args.output_path, args.dataset, root_path, args.png)
    else:
        raise ValueError("Invalid mode. Choose from 'entropy', 'lbp', or 'hybrid'.")
