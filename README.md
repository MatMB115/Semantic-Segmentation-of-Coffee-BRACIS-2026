# Coffee Field Segmentation Under Spatial Generalization

<p align="center">
  <img src="https://bracis.sbc.org.br/2026/wp-content/uploads/2026/01/logo_transparente-2048x960.png" alt="BRACIS 2026" height="90">
  <img src="https://iei.unifei.edu.br/wp-content/uploads/2022/04/simbolo_RGB.png" alt="UNIFEI" height="90">
  <img src="https://www.letras.ufmg.br/padrao_cms/imagens/eventos/icones/fapemig.png" alt="FAPEMIG" height="90">
</p>

Source code and example notebook for the BRACIS 2026 paper **“Semantic Segmentation of Coffee Fields Under Spatial Generalization: An Ablation Study with PlanetScope and Sentinel-2”**.

This repository provides the code used for single-model inference, ensemble evaluation, and statistical analysis associated with the experiments reported in the paper.

## Study Context

This repository supports a coffee field semantic segmentation study conducted in the Campo das Vertentes Geographical Indication (IGCV) region, Minas Gerais, Brazil. The goal of the study was to evaluate how different modeling choices affect coffee field segmentation when models are tested under spatial generalization, using a municipality-level hold-out protocol.

Two multispectral satellite data sources were considered in the experiments:

* **PlanetScope**, with higher spatial resolution and finer field-level spatial detail;
* **Sentinel-2**, with lower spatial resolution but broader public availability and spectral consistency.

## Evaluated Models

Three semantic segmentation architectures were evaluated:

* **UNet++**, a convolutional encoder-decoder architecture with nested skip connections designed to improve multiscale feature fusion.
* **MANet**, a convolutional segmentation model that incorporates attention mechanisms for contextual feature refinement.
* **SegFormer**, a transformer-based segmentation architecture with a hierarchical encoder and lightweight decoder. In this study, SegFormer was evaluated with a MiT-B2 encoder.

The convolutional models used ResNet-based encoders.

## Experimental Phases

The experiments were organized into four sequential phases:

### Phase 1: Loss Function Comparison

The first phase compared two loss functions commonly used in imbalanced segmentation tasks:

* **Dice Loss**
* **Tversky Loss**

### Phase 2: Patch Size Comparison

The second phase evaluated the effect of input patch size using different spatial dimensions:

* **128 × 128 pixels**
* **256 × 256 pixels**
* **512 × 512 pixels**

### Phase 3: Vegetation Index Comparison

The third phase evaluated the contribution of handcrafted vegetation indices added as extra input channels. The tested indices were:

* **EVI**: Enhanced Vegetation Index
* **GNDVI**: Green Normalized Difference Vegetation Index
* **NDRE**: Normalized Difference Red Edge Index

### Phase 4: Ensemble With Test-Time Augmentation

The fourth phase evaluated heterogeneous model combination through soft-voting ensemble inference with test-time augmentation. The ensemble averaged predicted class probabilities from selected trained models, using spatial transformations during inference to improve prediction stability.

## Evaluation and Statistical Analysis

Model performance was evaluated using standard segmentation metrics, including:

* **Precision**
* **Recall**
* **F1 score**
* **Intersection over Union (IoU)**

The reported analyses distinguish between dataset-level metrics and per-sample metrics. Statistical comparisons were performed using non-parametric tests over per-sample IoU distributions, including Friedman and Wilcoxon tests with p-value adjustment when applicable.


## Repository Contents

```text
.
├── inference.py
├── ensemble.py
├── stats.py
├── example_notebook.ipynb
├── README.md
├── LICENSE
└── requirements.txt
```

Main files:

* `inference.py`: runs inference for individual segmentation models, computes metrics, and saves prediction masks.
* `ensemble.py`: performs soft-voting ensemble inference with optional test-time augmentation.
* `stats.py`: runs non-parametric statistical analyses over experimental result files.
* `example_notebook.ipynb`: demonstrates the expected workflow and file organization.

## Supplementary Materials

The datasets, trained weights, prediction outputs, split metadata, and experimental result tables are archived in Zenodo:

https://doi.org/10.5281/zenodo.20722135

The Zenodo record should be used for citation because it contains the archived version of the supplementary materials associated with the paper.

## Installation

Python 3.11 or newer is recommended. The scripts can run on CPU, but inference is much faster with a CUDA-capable GPU and a compatible PyTorch installation.

### Ubuntu

Install Python, virtual environment support, and GDAL-related system packages used by `rasterio`:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip build-essential gdal-bin libgdal-dev
```

Create the environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Arch

Install Python and GDAL from the official repositories:

```bash
sudo pacman -Syu
sudo pacman -S python python-pip python-virtualenv base-devel gdal
```

Create the environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows

Install Python 3.11 or newer from `python.org` or with `winget`:

```powershell
winget install Python.Python.3.11
```

Open PowerShell in the repository directory, create the environment, and install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow scripts for the current user and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

`rasterio` usually installs from a prebuilt wheel on Windows. If pip tries to build it from source, install Microsoft C++ Build Tools and retry inside the activated environment.

### Manual Dependency List

If `requirements.txt` don't work, install the main dependencies manually:

```bash
python -m pip install numpy pandas scipy rasterio torch torchvision torchmetrics segmentation-models-pytorch ttach
```

## Expected Data Layout

The scripts expect the extracted Zenodo materials to follow a structure similar to:

```text
project_root/
├── code/
│   ├── inference.py
│   ├── ensemble.py
│   └── stats.py
├── dataset/
│   └── S2/
│       ├── P1/
│       ├── P2/
│       └── P3/
└── results/
    ├── S2/
    ├── PLANET/
    └── stats_analysis_results.json
```

Each dataset configuration follows the split structure:

```text
train/images/
train/masks/
val/images/
val/masks/
test/images/
test/masks/
metadata_split.csv
```

## Running Single-Model Inference

Linux/macOS example:

```bash
python inference.py \
  --data-root dataset/S2/P1/256 \
  --checkpoint-dir results/S2/P1_BASELINE/working_13_04_2026_05_09_40 \
  --output-dir results/reproduced/S2/P1_BASELINE \
  --per-sample-agg mean \
  --municipality-metrics
```

Windows PowerShell example:

```powershell
python inference.py `
  --data-root dataset\S2\P1\256 `
  --checkpoint-dir results\S2\P1_BASELINE\working_13_04_2026_05_09_40 `
  --output-dir results\reproduced\S2\P1_BASELINE `
  --per-sample-agg mean `
  --municipality-metrics
```

With test-time augmentation:

```bash
python inference.py \
  --data-root dataset/S2/P1/256 \
  --checkpoint-dir results/S2/P1_BASELINE/working_13_04_2026_05_09_40 \
  --output-dir results/reproduced/S2/P1_BASELINE_TTA \
  --tta
```

Arguments:

| Argument                 | Required | Default  | Description                                                                       |
| ------------------------ | -------- | -------- | --------------------------------------------------------------------------------- |
| `--data-root`            | Yes      | -        | Dataset root containing `test/images` and `test/masks`.                           |
| `--checkpoint-dir`       | Yes      | -        | Directory containing `best_*.pth` checkpoint files.                               |
| `--output-dir`           | Yes      | -        | Directory where inference outputs will be saved.                                  |
| `--tta`                  | No       | disabled | Enables test-time augmentation with D4 spatial transforms.                        |
| `--per-sample-agg`       | No       | `mean`   | Aggregation used to summarize per-sample metrics. Valid values: `mean`, `median`. |
| `--municipality-metrics` | No       | disabled | Computes and saves metrics grouped by municipality parsed from image names.       |

## Running Ensemble Inference

Linux/macOS example:

```bash
python ensemble.py \
  --data-root dataset/S2/P2/512 \
  --output-dir results/reproduced/S2/P4_ENSEMBLE_TTA \
  --ckpt-model-a results/S2/P4_ENSEMBLE_TTA/best_segformer_mit_b2_Dice.pth \
  --ckpt-model-b results/S2/P4_ENSEMBLE_TTA/best_unetpp_resnet34_Dice.pth \
  --weight-a 0.5 \
  --weight-b 0.5 \
  --tta
```

Windows PowerShell example:

```powershell
python ensemble.py `
  --data-root dataset\S2\P2\512 `
  --output-dir results\reproduced\S2\P4_ENSEMBLE_TTA `
  --ckpt-model-a results\S2\P4_ENSEMBLE_TTA\best_segformer_mit_b2_Dice.pth `
  --ckpt-model-b results\S2\P4_ENSEMBLE_TTA\best_unetpp_resnet34_Dice.pth `
  --weight-a 0.5 `
  --weight-b 0.5 `
  --tta
```

The ensemble script saves prediction masks, per-image results, and summary metrics in the selected output directory.

Arguments:

| Argument         | Required | Default  | Description                                                                |
| ---------------- | -------- | -------- | -------------------------------------------------------------------------- |
| `--data-root`    | Yes      | -        | Dataset root containing `test/images` and `test/masks`.                    |
| `--output-dir`   | Yes      | -        | Directory where ensemble outputs will be saved.                            |
| `--ckpt-model-a` | Yes      | -        | Path to the Model A checkpoint file (`.pth`).                              |
| `--ckpt-model-b` | Yes      | -        | Path to the Model B checkpoint file (`.pth`).                              |
| `--weight-a`     | No       | `0.5`    | Weight applied to Model A probabilities. Must be non-negative.             |
| `--weight-b`     | No       | `0.5`    | Weight applied to Model B probabilities. Must be non-negative.             |
| `--tta`          | No       | disabled | Enables test-time augmentation with D4 spatial transforms for both models. |

## Running Statistical Analysis

The statistical script expects a JSON configuration file describing the experimental phases, result CSV files, metrics, and statistical tests.

Linux/macOS example:

```bash
python stats.py experiment_config.example.json \
  --output results/reproduced/stats_analysis_results.json
```

Windows PowerShell example:

```powershell
python stats.py experiment_config.example.json `
  --output results\reproduced\stats_analysis_results.json
```

Each CSV referenced in the JSON must contain the selected metric column, such as `iou` or `f1`. For paired analyses, include an `image_id` column in every CSV so samples can be aligned safely across models.

The analyses used in the paper include non-parametric tests such as Friedman, Wilcoxon signed-rank, Kruskal-Wallis, and Mann-Whitney U, with optional p-value correction methods such as Holm and Bonferroni.

Arguments:

| Argument      | Required | Default                 | Description                           |
| ------------- | -------- | ----------------------- | ------------------------------------- |
| `config_json` | Yes      | -                       | Path to the JSON configuration file.  |
| `--output`    | No       | `analysis_results.json` | Path to the output JSON results file. |

## Citation

If you use this repository or the associated supplementary materials, please cite the Zenodo record and the BRACIS 2026 paper.

```text
Batista, M. M., Batista, B. G., Souza, V. C. O., Volpato, M. M. L., and Alves, H. M. R. Supplementary Materials for “Semantic Segmentation of Coffee Fields Under Spatial Generalization: An Ablation Study with PlanetScope and Sentinel-2”. Zenodo. https://doi.org/10.5281/zenodo.20722135
```

## License

The source code in this repository is released under the MIT License.

Supplementary materials associated with the paper, including Sentinel-2 patches, trained weights, split metadata, prediction outputs, and experimental results, are archived in Zenodo under the license specified in the Zenodo record.

PlanetScope imagery is not redistributed due to licensing restrictions.

## Contact

For questions about the code or supplementary materials, contact:

```text
Matheus M. Batista
Federal University of Itajubá (UNIFEI)
matmb@unifei.edu.br
```
