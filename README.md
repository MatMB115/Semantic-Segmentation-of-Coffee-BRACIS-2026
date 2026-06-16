# Coffee Field Segmentation Under Spatial Generalization

Source code and example notebook for the BRACIS 2026 paper **“Semantic Segmentation of Coffee Fields Under Spatial Generalization: An Ablation Study with PlanetScope and Sentinel-2”**.

This repository provides the code used for single-model inference, ensemble evaluation, and statistical analysis in a semantic segmentation study of coffee fields using multispectral remote sensing imagery. The experiments compare PlanetScope and Sentinel-2 data under a municipality-level spatial generalization protocol.

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

```text
https://doi.org/10.5281/zenodo.20722135
```

The Zenodo record should be used for citation because it contains the archived version of the supplementary materials associated with the paper.

PlanetScope imagery is not redistributed due to licensing restrictions. Sentinel-2 patches, rasterized masks, trained weights, split metadata, and experimental outputs are provided in the Zenodo record according to the licensing terms described there.

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

### Arch Linux

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
