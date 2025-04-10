# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms as T
from einops import rearrange
from matplotlib import colors
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torchgeo.datasets.utils import percentile_normalization
from mveo_benchmarks.dataset import DFC2022, Potsdam, Vaihingen, BaseDataset
from omegaconf import OmegaConf

DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class Preprocessor:
    """Preprocesses input data by normalizing RGB channels and the height/elevation data.
    
    This class handles normalization of RGB channels (dividing by 255) and normalizes 
    height data (DEM/DSM) to the range [0, 1] based on dataset-specific min/max values.
    """
    
    def __init__(self, extra_vmin: float, extra_vmax: float) -> None:
        """Initialize the preprocessor with min/max values for height data normalization.
        
        Args:
            extra_vmin: Minimum value for the height data (DEM/DSM)
            extra_vmax: Maximum value for the height data (DEM/DSM)
        """
        self.extra_vmin = extra_vmin
        self.extra_vmax = extra_vmax

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process the sample by normalizing its values.
        
        Args:
            sample: Dictionary containing image and mask tensors
            
        Returns:
            Processed sample with normalized values
        """
        # Normalize RGB channels to [0,1]
        sample["image"][:-1] /= 255.0
        
        # Normalize height channel (DEM or DSM) to [0,1]
        sample["image"][-1] = (sample["image"][-1] - self.extra_vmin) / (self.extra_vmax - self.extra_vmin)
        sample["image"][-1] = torch.clip(sample["image"][-1], min=0.0, max=1.0)
        
        if "mask" in sample:
            # Set class 15 (usually clouds/shadows) to background (0)
            sample["mask"][sample["mask"] == 15] = 0
            # Add channel dimension to mask
            sample["mask"] = rearrange(sample["mask"], "h w -> () h w")
        return sample


class BaseDataModule(pl.LightningDataModule):
    """Base data module for all geospatial datasets.
    
    This class handles the common functionality for loading and processing
    geospatial datasets, including data loading, transformation, and 
    visualization.
    """
    
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        dataset_class: Type[BaseDataset],
        dataset_config_name: str,
        augmentations: Optional[Any] = DEFAULT_AUGS,
        **kwargs,
    ):
        """Initialize the data module.
        
        Args:
            root: Path to dataset root directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_scores_file: Path to file containing training sample scores
            train_size: Number of training samples to use
            dataset_class: Class of the dataset to instantiate
            dataset_config_name: Name of dataset in config.yaml
            augmentations: Augmentations to apply during training
            **kwargs: Additional arguments
        """
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scores_file = train_scores_file
        self.train_size = train_size
        self.augmentations = augmentations
        self.dataset_class = dataset_class
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        
        # Load dataset-specific configuration
        config = OmegaConf.load("config.yaml")
        dataset_config = config["datasets"][dataset_config_name]
        
        # Get min/max values for height data normalization
        extra_min_key = "dem_min" if "dem_min" in dataset_config else "dsm_min"
        extra_max_key = "dem_max" if "dem_max" in dataset_config else "dsm_max"
        self.extra_min = dataset_config[extra_min_key]
        self.extra_max = dataset_config[extra_max_key]
        self.pad_size = dataset_config["pad_size"]
        self.preprocess = Preprocessor(self.extra_min, self.extra_max)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the dataset splits.
        
        Args:
            stage: Current stage (fit, validate, test, predict)
        """
        transforms = T.Compose([self.preprocess])

        self.train_ds = self.dataset_class(
            self.root,
            split="train",
            scores_file=self.train_scores_file,
            sample_size=self.train_size,
            transforms=transforms,
        )

        self.val_ds = self.dataset_class(
            self.root,
            split="val",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

        self.test_ds = self.dataset_class(
            self.root,
            split="test",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.
        
        Returns:
            DataLoader for training data
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.
        
        Returns:
            DataLoader for validation data
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.
        
        Returns:
            DataLoader for test data
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size * 4,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch: Dict[str, torch.Tensor], dl_idx: int) -> Dict[str, torch.Tensor]:
        """Apply augmentations after batch is moved to device.
        
        Args:
            batch: Batch of data
            dl_idx: Dataloader index
            
        Returns:
            Augmented batch
        """
        if self.trainer.training:
            if self.augmentations is not None:
                # Convert mask to float for augmentation compatibility
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                # Convert back to long for loss function compatibility
                batch["mask"] = batch["mask"].to(torch.long)
        # Remove channel dimension from mask
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(
        self,
        sample: Dict[str, torch.Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Visualize a sample with RGB, height map, and masks.
        
        Args:
            sample: Dictionary with image and mask tensors
            show_titles: Whether to show subplot titles
            suptitle: Super title for the figure
            
        Returns:
            Matplotlib figure with visualizations
        """
        ncols = 2

        image = sample["image"][:3]
        image = (image * 255.0).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(self.dataset_class.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(
                mask, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(
                    pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
                )
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(
                pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM" if hasattr(self, "dem_min") else "DSM")
            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class DFC2022DataModule(BaseDataModule):
    """Data module for DFC2022 dataset."""
    
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations: Optional[Any] = DEFAULT_AUGS,
        **kwargs,
    ):
        """Initialize the DFC2022 data module.
        
        Args:
            root: Path to dataset root directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_scores_file: Path to file containing training sample scores
            train_size: Number of training samples to use
            augmentations: Augmentations to apply during training
            **kwargs: Additional arguments
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_scores_file=train_scores_file,
            train_size=train_size,
            dataset_class=DFC2022,
            dataset_config_name="dfc22",
            augmentations=augmentations,
            **kwargs,
        )


class PotsdamDataModule(BaseDataModule):
    """Data module for Potsdam dataset."""
    
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations: Optional[Any] = DEFAULT_AUGS,
        **kwargs,
    ):
        """Initialize the Potsdam data module.
        
        Args:
            root: Path to dataset root directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_scores_file: Path to file containing training sample scores
            train_size: Number of training samples to use
            augmentations: Augmentations to apply during training
            **kwargs: Additional arguments
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_scores_file=train_scores_file,
            train_size=train_size,
            dataset_class=Potsdam,
            dataset_config_name="potsdam",
            augmentations=augmentations,
            **kwargs,
        )


class VaihingenDataModule(BaseDataModule):
    """Data module for Vaihingen dataset."""
    
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations: Optional[Any] = DEFAULT_AUGS,
        **kwargs,
    ):
        """Initialize the Vaihingen data module.
        
        Args:
            root: Path to dataset root directory
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            train_scores_file: Path to file containing training sample scores
            train_size: Number of training samples to use
            augmentations: Augmentations to apply during training
            **kwargs: Additional arguments
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            train_scores_file=train_scores_file,
            train_size=train_size,
            dataset_class=Vaihingen,
            dataset_config_name="vaihingen",
            augmentations=augmentations,
            **kwargs,
        )