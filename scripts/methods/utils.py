# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
import pandas as pd
import numpy as np


def entropy(v, eps=1e-6):
    """Calculate Shannon entropy of a probability distribution.
    
    Args:
        v: Probability distribution vector
        eps: Small value to avoid log(0)
        
    Returns:
        Entropy value
    """
    return -np.sum(v * np.log(v + 1e-6))


def get_samples(store=None, root_dir=None):
    """Extract sample metadata from filenames in a directory.
    
    Parses filenames in the format "{id}_{xmin}_{ymin}.tif" to extract
    metadata about each sample.
    
    Args:
        store: Path object pointing to the directory (alternative to root_dir)
        root_dir: String path to the directory (alternative to store)
        
    Returns:
        DataFrame with id, xmin, ymin, and score columns for each sample
    """
    if root_dir is not None:
        store = Path(root_dir)
    files = [f.stem for f in list(store.glob("*.tif"))]
    data = {"id": list(), "xmin": list(), "ymin": list(), "score": list()}
    for fn in files:
        # Parse filename to extract ID and coordinates
        info = fn.split("_")
        xmin, ymin = map(int, info[-2:])  # Last two parts are coordinates
        img_id = "_".join(info[:-2])      # Everything else is the ID
        data["id"].append(img_id)
        data["xmin"].append(xmin)
        data["ymin"].append(ymin)
        data["score"].append(1.0)         # Default score of 1.0
    return pd.DataFrame(data)

