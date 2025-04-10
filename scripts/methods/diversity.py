# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from sklearn import preprocessing
import scipy.linalg
from tqdm import tqdm

INTRA_CLUSTER_SAMPLING_METHOD = "random"
INTER_CLUSTER_SAMPLING_METHOD = "weighted"


def score_dual(X, q=1, normalize=True):
    """Calculate Vendi Score - a diversity metric based on eigenvalue distributions.
    
    The Vendi Score measures the diversity of image representations using entropy
    of eigenvalues from the similarity matrix.
    
    Args:
        X: Matrix of features
        q: Order of Renyi entropy (default=1 for Shannon entropy)
        normalize: Whether to normalize feature vectors
        
    Returns:
        Vendi Score value (higher means more diverse)
    """
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    # Compute similarity matrix
    S = X.T @ X
    # Calculate eigenvalues of the similarity matrix
    w = scipy.linalg.eigvalsh(S / n)
    
    def entropy_q(p, q=1):
        """Calculate q-entropy of a probability distribution."""
        p_ = p[p > 0]
        if q == 1:
            return -(p_ * np.log(p_)).sum()
        if q == "inf":
            return -np.log(np.max(p))
        return np.log((p_**q).sum()) / (1 - q)
    
    return np.exp(entropy_q(w, q=q))


class SimpleImageDataset(Dataset):
    """Dataset for loading and processing image patches."""
    
    def __init__(self, root_dir, transforms=None):
        """Initialize the dataset.
        
        Args:
            root_dir: Directory containing image files
            transforms: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.file_paths = []
        for file in Path(root_dir).glob('**/*.tif'):
            self.file_paths.append(str(file))        
        print(f"Found {len(self.file_paths)} images in {root_dir}")
    
    def __len__(self):
        """Get the number of images in the dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """Get an image by index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary with image tensor, index, and file path
        """
        file_path = self.file_paths[idx]
        with rasterio.open(file_path) as f:
            array = f.read(list(range(1, 4)))  # Get first 3 bands (RGB)
            image_tensor = torch.from_numpy(array.astype("float32"))
        
        if self.transforms is not None:
            if isinstance(self.transforms, Preprocessor):
                sample = {"image": image_tensor, "index": idx, "path": file_path}
                sample = self.transforms(sample)
                return sample
            else:
                image_tensor = self.transforms(image_tensor)
        
        return {"image": image_tensor, "index": idx, "path": file_path}


class Preprocessor(object):
    """Normalize images with ImageNet mean and standard deviation."""
    
    def __init__(self, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
        """Initialize with ImageNet normalization values.
        
        Args:
            means: Mean values for RGB channels
            stds: Standard deviation values for RGB channels
        """
        self.normalize = Normalize(means, stds)
    
    def __call__(self, sample):
        """Apply normalization to the image in the sample.
        
        Args:
            sample: Dictionary containing image tensor
            
        Returns:
            Processed sample with normalized image
        """
        if "image" in sample:
            sample["image"] = self.normalize(sample["image"])
        return sample


def get_mean_vendiscore(representations, labels):
    """Calculate the mean Vendi Score across different clusters.
    
    Args:
        representations: Matrix of feature vectors
        labels: Cluster labels for each feature vector
        
    Returns:
        Mean Vendi Score across all clusters
    """
    vendis = []
    for label in np.unique(labels):
        # Calculate Vendi Score for each cluster
        cur_vendi = score_dual(representations[labels == label, :])
        vendis.append(cur_vendi)
    return np.mean(vendis)


def model_config(arch="resnet"):
    """Configure feature extraction model based on architecture.
    
    Args:
        arch: Architecture name ('resnet' or 'vit')
        
    Returns:
        Tuple of (model, feature dimension, transforms)
    """
    if arch == "resnet":
        # Use ResNet18 pretrained on ImageNet
        base_model = models.resnet18(weights="DEFAULT")
        # Remove classification layer
        model = nn.Sequential(*list(base_model.children())[:-1])
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        feat_dim = 512
        transforms = Preprocessor()
    elif arch == "vit":
        # Use Vision Transformer pretrained on ImageNet-21k
        model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            num_classes=0,
        )
        model = model.eval()
        feat_dim = 768
        data_config = timm.data.resolve_data_config(args={}, model=model)
        base_transforms = timm.data.create_transform(**data_config, is_training=False)
        
        class VitPreprocessor(Preprocessor):
            def __call__(self, sample):
                if "image" in sample:
                    # Apply ViT-specific transforms then normalize
                    sample["image"] = base_transforms(sample["image"])
                    sample["image"] = self.normalize(sample["image"])
                return sample
        
        transforms = VitPreprocessor()
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(f"Model architecture '{arch}' not supported")
    return model, feat_dim, transforms


def extract_representations(dl, model, batch_size=8, feat_size=2048, arch="resnet"):
    """Extract feature representations from images using the model.
    
    Args:
        dl: DataLoader containing images
        model: Feature extraction model
        batch_size: Batch size
        feat_size: Feature dimension size
        arch: Architecture name ('resnet' or 'vit')
        
    Returns:
        Tuple of (representations, indices, file paths)
    """
    datasize = len(dl.dataset)
    representations = np.zeros((datasize, feat_size))
    indices = []
    paths = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i, batch in tqdm(enumerate(dl), total=len(dl)):
        
        image_tensor = batch["image"].to(device)
        batch_indices = batch["index"]
        batch_paths = batch["path"]
        
        indices.extend(batch_indices.tolist())
        paths.extend(batch_paths)
        
        with torch.no_grad():
            feats = model(image_tensor).detach().cpu().numpy()
        
        start_pos = i * batch_size
        if arch == "vit":
            # ViT features are already flattened
            representations[start_pos : start_pos + len(batch_indices), :] = feats
        elif arch == "resnet":
            # Flatten spatial dimensions for ResNet features
            representations[start_pos : start_pos + len(batch_indices), :] = feats[:, :feat_size, 0, 0]
        else:
            raise Exception(f"Model architecture '{arch}' not supported")
    
    return representations, indices, paths


def cluster_representations(X, n_clusters):
    """Cluster representations using K-means.
    
    Args:
        X: Matrix of feature vectors
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels for each feature vector
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.labels_


def find_optimal_cluster(X):
    """Find the optimal number of clusters based on Vendi Score changes.
    
    This function iteratively increases the number of clusters until the
    mean change in Vendi Score becomes small, indicating diminishing returns.
    
    Args:
        X: Matrix of feature vectors
        
    Returns:
        Optimal number of clusters
    """
    # Start with one cluster (all samples together)
    mean_vendi_score = score_dual(X)
    all_vendi_scores = [mean_vendi_score]
    vendi_change = np.inf
    mean_vendi_change = np.inf
    cur_mean_vendi = mean_vendi_score
    clusters = 2  # Start with 2 clusters

    overall_vendi_change = []
    # Continue until the mean Vendi Score change becomes small
    while mean_vendi_change > 0.005:
        print(f"Number of clusters used: {clusters} with mean vendi score: {cur_mean_vendi}")
        cluster_model = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(X)
        cur_mean_vendi = get_mean_vendiscore(X, cluster_model.labels_)
        vendi_change = abs(mean_vendi_score - cur_mean_vendi) / cur_mean_vendi
        overall_vendi_change.append(vendi_change)
        if cur_mean_vendi < mean_vendi_score:
            mean_vendi_score = cur_mean_vendi
        all_vendi_scores.append(cur_mean_vendi)
        clusters += 1
        # Use rolling average of last 3 changes to smooth out fluctuations
        mean_vendi_change = np.mean(overall_vendi_change[-3:])
    return clusters


def map_clusters_to_samples(embeddings, paths, cluster_labels, indices):
    """Map cluster information to samples and calculate diversity scores.
    
    Args:
        embeddings: Matrix of feature vectors
        paths: List of file paths
        cluster_labels: Cluster label for each sample
        indices: Original indices of samples
        
    Returns:
        DataFrame with cluster assignments and diversity scores
    """
    df = pd.DataFrame({'path': paths, 'index': indices})
    df['filename'] = df['path'].apply(lambda x: os.path.basename(x))
    df["clusters"] = None
    df["diversity_score"] = None
    
    # Assign cluster labels and calculate diversity scores
    for label in np.unique(cluster_labels):
        index_in_embeddings = np.where(cluster_labels == label)[0].tolist()
        positions_in_df = [indices[k] for k in index_in_embeddings]
        df.loc[df['index'].isin(positions_in_df), "clusters"] = label

        # Calculate diversity score for this cluster
        diversity_score = score_dual(embeddings[index_in_embeddings, :])
        df.loc[df['index'].isin(positions_in_df), "diversity_score"] = diversity_score
    
    # Normalize diversity scores to integers starting at 1
    min_score = df["diversity_score"].min()
    if min_score > 0:
        df["diversity_score"] /= min_score
        df["diversity_score"] = df["diversity_score"].astype(int)
    
    return df


def build_cluster_dict(df, intra_cluster_sampling_method=INTRA_CLUSTER_SAMPLING_METHOD):
    """Build a dictionary of samples by cluster with optional sorting.
    
    Args:
        df: DataFrame with cluster assignments
        intra_cluster_sampling_method: Method to order samples within clusters
        
    Returns:
        Tuple of (cluster dictionary, cluster sizes, diversity scores dictionary)
    """
    cluster_dict = {}
    cluster_sizes = []
    diversity_score_dict = {}
    
    for label, group in df.groupby("clusters"):
        if intra_cluster_sampling_method == "random":
            # Randomly shuffle samples within each cluster
            tmp_list = group.sample(frac=1).index.to_list()
        elif intra_cluster_sampling_method == "sample_selection_score":
            # Sort by sample selection score
            tmp_list = group.sort_values(by="sample_selection_score", ascending=False).index.to_list()
        else:
            raise ValueError(f"Unsupported intra_cluster_sampling_method: {intra_cluster_sampling_method}")
            
        diversity_score_dict[label] = group["diversity_score"].mean()
        cluster_dict[label] = tmp_list
        cluster_sizes.append(len(tmp_list))

    return cluster_dict, cluster_sizes, diversity_score_dict


def sample_from_clusters(
    cluster_dict, cluster_sizes, diversity_score_dict, inter_cluster_sampling_method=INTER_CLUSTER_SAMPLING_METHOD
):
    """Sample from clusters in a round-robin fashion with optional weighting.
    
    Args:
        cluster_dict: Dictionary mapping cluster labels to sample indices
        cluster_sizes: List of cluster sizes
        diversity_score_dict: Dictionary mapping cluster labels to diversity scores
        inter_cluster_sampling_method: Method for inter-cluster sampling
        
    Returns:
        List of sample indices in the desired order
    """
    full_list = []
    max_cluster_size = max(cluster_sizes)
    
    for el_pos in range(max_cluster_size):
        for k in cluster_dict.keys():
            current_cluster = cluster_dict[k]
            
            if inter_cluster_sampling_method == "weighted":
                # Sample more from diverse clusters
                sample_weight = int(diversity_score_dict[k])
            elif inter_cluster_sampling_method == "unweighted":
                # Sample equally from all clusters
                sample_weight = 1
            else:
                raise Exception(f"Inter cluster sampling method '{inter_cluster_sampling_method}' not supported")

            # For each position, take multiple samples from the cluster based on its weight
            if (el_pos * sample_weight) < len(current_cluster):
                start_pos = el_pos * sample_weight
                end_pos = start_pos + sample_weight
                full_list.append(current_cluster[start_pos:end_pos])
                
    # Flatten the list of lists
    full_list = [item for sublist in full_list for item in sublist]
    return full_list


def create_ranking_by_df(df, order_list, output_path):
    """Create a ranked file based on the specified order.
    
    Args:
        df: DataFrame with sample information
        order_list: List of sample indices in the desired order
        output_path: Path to save the output CSV file
    """
    df["final_scores"] = None
    list_size = len(order_list)
    
    # Assign scores from 1 to 0 based on position in order_list
    for idx, i in enumerate(order_list):
        df.loc[i, "final_scores"] = (list_size - idx) / list_size

    output_df = df[["filename", "final_scores", "clusters"]]
    output_df.rename(columns={"final_scores": "score"}, inplace=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df[['id','xmin','ymin']] = output_df['filename'].apply(
        lambda x: (
            '_'.join(x[:-4].split('_')[:3]),  
            x[:-4].split('_')[3],              
            x[:-4].split('_')[4]               
        )
    ).tolist()

    output_df['xmin'] = output_df['xmin'].astype(int)
    output_df['ymin'] = output_df['ymin'].astype(int)
    output_df[['id','xmin','ymin','score']].to_csv(output_path)
    print(f"Saved rankings to {output_path}")


def get_image(file_path):
    """Load an image from file for visualization.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        RGB image array normalized to [0,1]
    """
    with rasterio.open(file_path) as f:
        array = f.read()
    return np.rollaxis(array[:3, :, :], 0, 3) / 255.0


def plot_clusters(df, clusters_png="clusters.png"):
    """Visualize representative samples from each cluster.
    
    Args:
        df: DataFrame with cluster assignments
        clusters_png: Path to save the visualization
    """
    num_samples = min(10, df.groupby('clusters').size().min())
    sample_labels = df.clusters.nunique()
    
    _, axs = plt.subplots(
        sample_labels, num_samples, figsize=(num_samples*2, sample_labels*2)
    )
    if sample_labels == 1:
        axs = np.expand_dims(axs, axis=0)
    
    for cluster, group in df.groupby("clusters"):
        cur_samples = group.sample(min(num_samples, len(group)))
        cur_samples = cur_samples.reset_index()
        
        for idx, row in cur_samples.iterrows():
            img = get_image(row["path"])
            ax = axs[int(cluster), idx]
            ax.imshow(img)
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel(f"Cluster {int(cluster)}")
    
    plt.tight_layout()
    plt.savefig(clusters_png, dpi=100, bbox_inches='tight', pad_inches=0.1, format='png')
    print(f"Saved cluster visualization to {clusters_png}")


def main(args):
    """Main function to run the diversity-based ranking pipeline.
    
    Args:
        args: Command-line arguments
    """
    batch_size = 64
    num_workers = 16
    model, feat_size, transform = model_config(arch=args.arch)
    ds = SimpleImageDataset(
        root_dir=args.root_dir,
        transforms=transform
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    print("Extracting image representations...")
    embeddings, indices, paths = extract_representations(
        dl,
        model,
        batch_size=batch_size,
        feat_size=feat_size,
        arch=args.arch,
    )

    # Sample a subset for efficiency in finding optimal clusters
    sample_embeddings = embeddings[np.random.choice(embeddings.shape[0], size=int(embeddings.shape[0] * 0.1), replace=False), :]
    print("Finding optimal number of clusters...")
    n_clusters = find_optimal_cluster(sample_embeddings)
    print(f"Optimal number of clusters: {n_clusters}")
    print("Clustering representations...")
    cluster_labels = cluster_representations(embeddings, n_clusters)
    print("Mapping clusters to samples...")
    cluster_samples_df = map_clusters_to_samples(
        embeddings, paths, cluster_labels, indices
    )
    print("Plotting clusters...")
    plot_clusters(
        cluster_samples_df,
        clusters_png=args.clusters_png,
    )
    print("Building cluster dictionary...")
    cluster_dict, cluster_sizes, diversity_score_dict = build_cluster_dict(
        cluster_samples_df, INTRA_CLUSTER_SAMPLING_METHOD
    )
    print("Sampling from clusters...")
    ranking_list = sample_from_clusters(
        cluster_dict,
        cluster_sizes,
        diversity_score_dict,
        INTER_CLUSTER_SAMPLING_METHOD,
    )
    print("Creating final rankings...")
    create_ranking_by_df(cluster_samples_df, ranking_list, args.output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image clustering and ranking")
    
    parser.add_argument(
        "--root_dir",
        required=True,
        type=str,
        help="Directory containing the image files"
    )
    
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save the output CSV file with scores"
    )
    
    parser.add_argument(
        "--arch",
        required=True,
        type=str,
        choices=["resnet", "vit"],
        help="Architecture to use for feature extraction"
    )
    
    parser.add_argument(
        "--clusters_png",
        required=True,
        type=str,
        help="Path to save the PNG file with cluster examples"
    )

    args = parser.parse_args()
    main(args)
