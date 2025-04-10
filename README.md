# Data-Centric Methods for Satellite Image Segmentation

This repository contains implementations of data-centric approaches for improving semantic segmentation performance on satellite imagery. These methods won the [MVEO data-centric competition](https://mveo.github.io/).

## Overview

We share the implementation of techniques for prioritizing training samples based on different measures:
- **Diversity-based selection**: Prioritizing samples that represent the diversity of the dataset
- **Complexity-based ranking**: Focusing on samples with higher information content

The main dataset supported is `DFC-22`, with additional experimental support for Potsdam and Vaihingen datasets.

## Setup

Create an environment as follows:

```bash
mamba create -n mveo python=3.12.3
conda activate mveo
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # [Optional]
```

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
pip install -e .
```

Create the directories structure where raw and prepared data will be stored:
```
root/
├── raw/
│   ├── dfc22/
│   ├── vaihingen/
│   └── potsdam/
├── dfc22/
│   ├── train/
│   ├── val/
│   └── test/
├── vaihingen/
│   ├── train/
│   ├── val/
│   └── test/
└── potsdam/
    ├── train/
    ├── val/
    └── test/
```

Set the absolute path to the root directory at `./config.yaml`.

## DFC-22 Dataset

### Data Acquisition

Go to the IEEE Dataport: https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022#files

.. and left-click copy the URLs for `labeled train`, `validation`, and `test. 

`cd` to the `raw` directory and download + extract the datasets using the URLs as follows:

```bash
curl -L -o train.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/labeled_train.zip?XXX"
unzip train.zip
mv labeled_train/ train/

curl -L -o val.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/val.zip?Y"
unzip val.zip

curl -L -o test.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/test.zip?Z"
unzip test.zip
mkdir -p test; unzip test.zip -d test
```

### Patch Extraction

For the test dataset, you need to do the following:
1. Acquire the test reference dataset (not publicly available). You can request them by emailing Ronny Hänsch (*rww.haensch@gmail.com*).
2. Download the zip file and extract the masks to `{root}/raw/dfc22/test/{city}/UrbanAtlas`.
3. Rename the mask files from `*_reference.tif` to `*_UA2012.tif`.

Go to the scripts directory:
```bash
cd scripts/extract_patches/
```

To export the train, validation, and test patches, run the following:

```bash
python dfc22.py \
    --indices_file ../../data/indices/dfc2022_train_val_test.csv \
    --source_dir {root}/raw/dfc22 \
    --output_dir {root}/dfc22
```

## Ranking Methods

Now, we want to use our methods to rank the patches for training.

### Random 

To establish a baseline (random) submission file, run the following:

```bash
# Random
python scripts/methods/baseline.py --root_dir {root}/dfc22/train --output_path ./data/submissions/random.csv --mode random

# Censored
python scripts/methods/baseline.py --root_dir {root}/dfc22/train --output_path ./data/submissions/random_censored.csv --mode censored

# Censored Balanced
python scripts/methods/baseline.py --root_dir {root}/dfc22/train --output_path ./data/submissions/random_censored_balanced.csv --mode balanced
```

### Diversity 
For diversity based ranking, run the following:

```bash
# Use ResNet Embeddingss
python scripts/methods/diversity.py \
    --root_dir {root}/dfc22/train \
    --arch resnet \
    --output_path ./data/submissions/diversity_resnet.csv \
    --clusters_png ./data/submissions/clusters_resnet.png

# Use ViT Embeddings
python scripts/methods/diversity.py \
    --root_dir {root}/dfc22/train \
    --arch vit \
    --output_path ./data/submissions/diversity_vit.csv \
    --clusters_png ./data/submissions/clusters_vit.png
```

### Complexity

For complexity based ranking, run the following:

```bash
# Entropy complexity
python scripts/methods/complexity.py \
    --root_dir {root} \
    --dataset dfc22 \
    --mode entropy \
    --output_path ./data/submissions/complexity_entropy.csv \
    --png ./data/submissions/complexity_entropy.png

# Local Binary Pattern
python scripts/methods/complexity.py \
    --root_dir {root} \
    --dataset dfc22 \
    --mode lbp \
    --output_path ./data/submissions/complexity_lbp.csv \
    --png ./data/submissions/complexity_lbp.png

# Hybrid approach
python scripts/methods/complexity.py \
    --root_dir {root} \
    --dataset dfc22 \
    --mode hybrid \
    --output_path ./data/submissions/complexity_hybrid.csv \
    --png ./data/submissions/complexity_hybrid.png
```

## Training

Launch training for DFC-22 as follows:

```bash
python scripts/train.py \
    --dataset "dfc22" \
    --method_name "DFC22Random" \
    --scores_file_path {root}/data/submissions/random.csv \
    --gpu 0
```

## Evaluation

For each run, jaccard scores for each class are saved. At the end of training, you will find all of the relevant scores saved in `./results/{method_name}.txt`.

Given you have the path to the best model checkpoint, you can also evaluate using the original images in `notebooks/export_results.ipynb`.

---

## Experimental Datasets

In addition to the main DFC-22 dataset, our methods can also be applied to the following experimental datasets.

### Potsdam

#### Data Acquisition

Visit [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)'s [link](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/) then left-click-copy the URL:

You can extract the ZIP file:
```bash
curl -L -o "Potsdam.zip" "https://seafile.projekt.uni-hannover.de/seafhttp/files/{KEY}/Potsdam.zip"
unzip Potsdam.zip
```

Then extract all compressed files in `Potsdam`:

```bash
cd scripts/data_preparation
chmod +x extract_files.sh
./extract_files.sh raw/potsdam/Potsdam
```

#### Patch Extraction

```bash
# Train
python potsdam.py \
    --indices_file ../../data/indices/potsdam_train_coordinate_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/train

# Validation
python potsdam.py \
    --indices_file ../../data/indices/potsdam_val_image_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/val

# Testing
python potsdam.py \
    --indices_file ../../data/indices/potsdam_test_image_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/test
```

#### Training

```bash
python scripts/train.py \
        --dataset "potsdam" \
        --method_name "PotsdamDiversity" \
        --scores_file_path {root}/submissions/potsdam/diversity.csv \
        --gpu 7
```

### Vaihingen

#### Data Acquisition

Visit [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)'s [link](https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/) to download the dataset.

You can extract the ZIP file:
```bash
unzip Vaihingen.zip
```

Then extract all compressed files in `Vaihingen`:

```bash
cd scripts/data_preparation
chmod +x extract_files.sh
./extract_files.sh raw/vaihingen/Vaihingen
```

#### Patch Extraction

```bash
# Train
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_train_coordinate_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/train

# Validation
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_val_image_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/val

# Testing
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_test_image_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/test
```

#### Training

```bash
python scripts/train.py \
        --dataset "vaihingen" \
        --method_name "VaihingenDiversity" \
        --scores_file_path {root}/submissions/vaihingen/diversity.csv \
        --gpu 0
```

## Data Attribution

This repository uses the following datasets:

### DFC-22 Dataset
The Data Fusion Contest 2022 (DFC-22) dataset is provided by IEEE GRSS, Université Bretagne-Sud, ONERA, and ESA Φ-lab. 

If you use this data, please cite:
1. 2022 IEEE GRSS Data Fusion Contest. Online: https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/
2. Castillo-Navarro, J., Le Saux, B., Boulch, A. and Lefèvre, S.. Semi-supervised semantic segmentation in Earth Observation: the MiniFrance suite, dataset analysis and multi-task network study. Mach Learn (2021). https://doi.org/10.1007/s10994-020-05943-y
3. Hänsch, R.; Persello, C.; Vivone, G.; Castillo Navarro, J.; Boulch, A.; Lefèvre, S.; Le Saux, B. : 2022 IEEE GRSS Data Fusion Contest: Semi-Supervised Learning [Technical Committees], IEEE Geoscience and Remote Sensing Magazine, March 2022

#### Usage conditions
The data are provided for research purposes and must be identified as "grss_dfc_2022" in any scientific publication.

### ISPRS Vaihingen Dataset
The Vaihingen dataset is part of the ISPRS 2D Semantic Labeling Benchmark. If you use this data, please cite:
- Cramer, M., 2010. The DGPF test on digital aerial camera evaluation – overview and test design. Photogrammetrie – Fernerkundung – Geoinformation 2(2010):73-82.

And include the following acknowledgement:
"The Vaihingen data set was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF) [Cramer, 2010]: http://www.ifp.uni-stuttgart.de/dgpf/DKEP-Allg.html."

#### Usage conditions
1. The data must not be used for other than research purposes. Any other use is prohibited.
2. The data must not be distributed to third parties. Any person interested in the data may obtain them via ISPRS WG III/4.
3. The German Association of Photogrammetry, Remote Sensing and GeoInformation (DGPF) should be informed about any published papers whose results are based on the Vaihingen test data.

### ISPRS Potsdam Dataset
The Potsdam dataset is part of the ISPRS 2D Semantic Labeling Benchmark. If you use this data, please cite:
- ISPRS 2D Semantic Labeling - Potsdam: https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

The dataset consists of 38 patches of true orthophotos (TOP) and digital surface models (DSM) with a ground sampling distance of 5 cm. The data is provided in different channel compositions (IRRG, RGB, RGBIR) as TIFF files.

#### Usage conditions
Based on similar ISPRS test datasets, this data is intended for research purposes only and should not be redistributed. Researchers interested in the data should obtain it directly from the ISPRS benchmark website.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## License

This project is licensed under the [MIT License](LICENSE).