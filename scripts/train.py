# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Training script for semantic segmentation models.

This script handles training and evaluation of semantic segmentation models on 
various remote sensing datasets. It supports training with different sample sizes
and multiple runs for robust evaluation.
"""
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pandas as pd

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mveo_benchmarks.tasks import SemanticSegmentationTask
from mveo_benchmarks.datamodule import (
    DFC2022DataModule,
    PotsdamDataModule,
    VaihingenDataModule,
)


def main(config):
    """
    Train and evaluate semantic segmentation models with various sample sizes.
    
    Args:
        config: Configuration dictionary loaded from yaml file containing
               parameters for training, evaluation, and dataset settings.
    """
    # Select the appropriate data module based on dataset name
    dataset = config["datamodule"]["dataset"]
    if dataset == "dfc22":
        dm = DFC2022DataModule
    elif dataset == "potsdam":
        dm = PotsdamDataModule
    elif dataset == "vaihingen":
        dm = VaihingenDataModule
    else:
        raise ValueError("Unknown dataset")

    # Load dataset-specific configurations
    dataset_config = config["datasets"][dataset]
    config["learning"]["in_channels"] = dataset_config["in_channels"]
    config["learning"]["ignore_index"] = dataset_config["ignore_index"]
    config["learning"]["num_classes"] = dataset_config["num_classes"]

    scores = {}
    for sample_size in config["evaluation"]["sizes"]:
        scores[sample_size] = []
        for run in range(config["evaluation"]["runs"]):
            # Initialize data module with current sample size
            datamodule = dm(
                **config.datamodule,
                train_size=sample_size,
                train_scores_file=config["evaluation"]["scores_file"],
            )
            task = SemanticSegmentationTask(**config.learning)
            ckpt_dir_path = f"checkpoints/{config['evaluation']['method_name'].lower().replace(' ', '_')}_{sample_size}_run{run}"

            # Setup callbacks for early stopping and model checkpointing
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=0.00, patience=20, mode="min"
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=ckpt_dir_path,
                filename="best_model",
                save_top_k=1,
                mode="min",
            )
            logger = pl.loggers.TensorBoardLogger(
                "logs/",
                name=f"{config['evaluation']['method_name'].lower().replace(' ', '_')}_{sample_size}_run{run}",
            )
            trainer = pl.Trainer(
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=logger,
                **config.trainer,
            )
            trainer.fit(model=task, datamodule=datamodule)
            
            # Test the model using the best checkpoint
            test_metrics = trainer.test(
                model=task,
                datamodule=datamodule,
                ckpt_path=checkpoint_callback.best_model_path,
            )
            
            # Extract and store Jaccard scores
            jaccard_scores = test_metrics[0]
            jaccard_scores.pop("test_loss")
            keys = list(jaccard_scores.keys())
            scores[sample_size].append([jaccard_scores[key] for key in keys])
            
            # Clean up resources
            torch.cuda.empty_cache()
            del task, datamodule, trainer

    # Write results to file
    scores_file_path = (
        Path("./results")
        / f"{config['evaluation']['method_name'].replace(' ', '_')}.txt"
    )
    with open(scores_file_path, "w") as file:
        for size, metrics in scores.items():
            metrics_str = ", ".join(map(str, metrics))
            file.write(f"{size}, {metrics_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["vaihingen", "potsdam", "dfc22"],
        type=str,
        help="The name of the scoring method.",
    )
    parser.add_argument(
        "--method_name", required=True, type=str, help="The name of the scoring method."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--scores_file_path",
        required=True,
        type=str,
        help="Path to scores file (similar to `train_ids.txt`)",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config_file_path)
    config["evaluation"]["method_name"] = args.method_name
    config["evaluation"]["scores_file"] = Path(args.scores_file_path)
    config["trainer"]["devices"] = [args.gpu] if args.gpu >= 0 else -1
    config["datamodule"]["root"] = Path(config["datamodule"]["root"]) / args.dataset
    config["datamodule"]["dataset"] = args.dataset

    # Convert percentage values to actual sample counts
    evaluation_percentages = config["evaluation"]["sizes"]
    scores = pd.read_csv(args.scores_file_path, sep=" ", header=None)
    n_samples = len(scores)
    config["evaluation"]["sizes"] = [
        int(n_samples * percentage) for percentage in evaluation_percentages
    ]

    main(config)
