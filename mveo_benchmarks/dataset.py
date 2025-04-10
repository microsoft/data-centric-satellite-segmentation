# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union


class BaseDataset(Dataset, ABC):
    """Base dataset class for all geospatial datasets.
    
    This abstract class provides the common infrastructure for loading
    geospatial data from TIF files, with support for scoring-based
    sample selection.
    """
    
    @property
    @abstractmethod
    def classes(self) -> List[str]:
        """List of class names in the dataset."""
        pass

    @property
    @abstractmethod
    def colormap(self) -> List[str]:
        """RGB hex color values for visualizing each class."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, str]:
        """Dictionary mapping split names to directory names."""
        pass

    def __init__(
        self, 
        root: Union[str, Path], 
        split: str, 
        scores_file: Optional[str] = None, 
        sample_size: Optional[int] = None, 
        transforms: Optional[Any] = None, 
        embed_fp: bool = False
    ) -> None:
        """Initialize the dataset.
        
        Args:
            root: Root directory containing the dataset
            split: Dataset split to use (train, val, test)
            scores_file: Path to file containing sample scores for prioritization
            sample_size: Number of samples to use (selects top-scored if scores_file is provided)
            transforms: Transforms to apply to samples
            embed_fp: Whether to include the file path in the returned sample
        """
        assert split in self.metadata
        self.root = Path(root)
        self.split = split
        self.scores_file = scores_file
        self.sample_size = sample_size
        self.transforms = transforms
        self.embed_fp = embed_fp
        self.files = self._load_files()

    def _load_files(self):
        """Load file paths, optionally prioritizing based on scores.
        
        Returns:
            List of file paths to use
        """
        if self.scores_file is not None:
            # When a scores file is provided, load and sort patches by score
            scores = self._parse_scores_file()
            patch_files = list()
            for _, row in scores.iterrows():
                id, xmin, ymin, score = (
                    row["id"],
                    int(row["xmin"]),
                    int(row["ymin"]),
                    float(row["score"]),
                )
                file_path = self.root / self.split / f"{id}_{xmin}_{ymin}.tif"
                patch_files.append((file_path, float(score)))
            # Sort by score in descending order
            patch_files = sorted(patch_files, key=lambda e: e[1], reverse=True)
            # Limit to sample_size if specified
            patch_files = (
                patch_files[: self.sample_size]
                if self.sample_size is not None
                else patch_files
            )
            patch_files = [file for file, _ in patch_files]
        else:
            # Without scores file, use all files in the split directory
            patch_files = list(Path(self.root).glob(f"{self.split}/*.tif"))
        return patch_files

    def _parse_scores_file(self) -> pd.DataFrame:
        """Parse the scores file.
        
        Returns:
            DataFrame containing patch IDs and their scores
        """
        scores = pd.read_csv(self.scores_file, index_col=0)
        return scores

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            Dictionary containing image and mask tensors
        """
        file_path = self.files[index]
        image, mask = self._load_data(file_path)
        sample = {"image": image, "mask": mask}
        if self.embed_fp:
            sample["fp"] = str(file_path.absolute())
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.files)

    def _load_data(self, path: Path, shape: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image and mask data from a file.
        
        Args:
            path: Path to the file
            shape: Optional shape to resize the data to
            
        Returns:
            Tuple of (image, mask) tensors
        """
        with rasterio.open(path) as f:
            arr = f.read()
            # Last channel is the mask, all others are image bands
            image_tensor = torch.from_numpy(arr[:-1, :, :].astype("float32"))
            mask_tensor = torch.from_numpy(arr[-1, :, :].astype("int32")).to(torch.long)
            return image_tensor, mask_tensor


class DFC2022(BaseDataset):
    """Dataset class for the DFC2022 competition.
    
    This dataset contains land cover classification data with 15 classes.
    """
    
    classes = [
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    ]

    colormap = [
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }


class Vaihingen(BaseDataset):
    """Dataset class for the ISPRS Vaihingen dataset.
    
    This dataset contains aerial imagery for urban land cover classification
    with 6 classes.
    """
    
    classes = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background",
    ]

    colormap = [
        "#FFFFFF",
        "#0000FF",
        "#00FFFF",
        "#00FF00",
        "#FFFF00",
        "#FF0000",
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }

    def _parse_scores_file(self) -> pd.DataFrame:
        """Parse the scores file for Vaihingen dataset.
        
        Handles the specific format of the Vaihingen scores file.
        
        Returns:
            DataFrame with parsed scores
        """
        scores = pd.read_csv(self.scores_file, sep=" ", header=None)
        scores = scores.iloc[:, [0, -3, -2, -1]]
        scores.columns = ["id", "xmin", "ymin", "score"]
        # Extract tile ID from the filename
        scores["id"] = scores["id"].apply(lambda x: x.split("_")[3])
        return scores


class Potsdam(BaseDataset):
    """Dataset class for the ISPRS Potsdam dataset.
    
    This dataset contains high-resolution aerial imagery for urban land cover
    classification with 6 classes.
    """
    
    classes = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background",
    ]

    colormap = [
        "#FFFFFF",
        "#0000FF",
        "#00FFFF",
        "#00FF00",
        "#FFFF00",
        "#FF0000",
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }

    def _parse_scores_file(self) -> pd.DataFrame:
        """Parse the scores file for Potsdam dataset.
        
        Handles the specific format of the Potsdam scores file.
        
        Returns:
            DataFrame with parsed scores
        """
        scores = pd.read_csv(self.scores_file, sep=" ", header=None)
        scores = scores.iloc[:, [0, -3, -2, -1]]
        scores.columns = ["id", "xmin", "ymin", "score"]
        # Extract tile ID from the filename
        scores["id"] = scores["id"].apply(lambda x: "_".join(x.split("_")[2:4]))
        return scores
