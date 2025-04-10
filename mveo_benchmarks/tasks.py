# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Segmentation tasks."""

import warnings
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchgeo.datasets.utils import unbind_samples
from torchgeo.models import FCN


class SemanticSegmentationTask(LightningModule):
    """Lightning module for semantic segmentation.
    
    This module supports different segmentation architectures (UNet, DeepLabV3+, FCN)
    and loss functions (CrossEntropy, Jaccard, Focal) for training semantic 
    segmentation models on geospatial data.
    """
    
    def config_task(self) -> None:
        """Configure the model architecture and loss function based on hyperparameters."""
        if self.hyperparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights=self.hyperparams["weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights=self.hyperparams["weights"],
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
                num_filters=self.hyperparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["loss"] == "ce":
            # Set ignore_index for CrossEntropyLoss to an extreme value if None
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.loss = nn.CrossEntropyLoss(ignore_index=ignore_value, reduction="none")
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hyperparams["num_classes"]
            )
        elif self.hyperparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the semantic segmentation task.
        
        Args:
            **kwargs: Hyperparameters for the task
        """
        super().__init__()

        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]
        self.config_task()

        # Setup metrics for training, validation, and testing
        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average="macro",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average="macro",
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = MetricCollection(
            [
                MulticlassJaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average=None,
                )
            ],
            prefix="test_",
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model.
        
        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            Model output
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Perform one training step.
        
        Args:
            *args: Batch and other arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Loss tensor
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        # Calculate loss using nanmean to handle ignore_index pixels
        loss = self.loss(y_hat, y).nanmean()

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def on_train_epoch_end(self) -> None:
        """Log metrics at the end of the training epoch."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Perform one validation step.
        
        Args:
            *args: Batch and batch_idx
            **kwargs: Additional keyword arguments
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        # Calculate loss using nanmean to handle ignore_index pixels
        loss = self.loss(y_hat, y).nanmean()

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        # Visualize predictions for first few batches
        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except ValueError:
                pass

    def on_validation_epoch_end(self) -> None:
        """Log metrics at the end of the validation epoch."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Perform one test step.
        
        Args:
            *args: Batch
            **kwargs: Additional keyword arguments
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y).nanmean()

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        """Log metrics at the end of the test epoch.
        
        Reports metrics for each class separately.
        """
        test_metrics = self.test_metrics.compute()

        for metric_name, metric_value in test_metrics.items():
            # For per-class metrics, log each class separately
            if isinstance(metric_value, torch.Tensor) and metric_value.numel() > 1:
                for i, val in enumerate(metric_value):
                    self.log(f"{metric_name}_class_{i}", val)
            else:
                self.log(metric_name, metric_value)
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Perform one prediction step.
        
        Args:
            *args: Batch
            **kwargs: Additional keyword arguments
            
        Returns:
            Softmax predictions
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }
