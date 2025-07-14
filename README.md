# YOLOv10 / YOLOv12 Training & Analysis Suite

**Author: Valerio Marchetti**

Complete suite for training and analyzing YOLOv10 / YOLOv12 models with support for ablation studies, CO2 emissions tracking, TensorBoard integration and visualization tools.

## Features

- ✅ YOLOv10/YOLOv12 training with Ultralytics
- ✅ Systematic ablation studies for model optimization
- ✅ CO2 emissions tracking with CodeCarbon
- ✅ Real-time TensorBoard monitoring
- ✅ Wandb integration for experiment management
- ✅ Pre-trained weights support
- ✅ Automatic model testing
- ✅ Detailed logging
- ✅ GPU memory management
- ✅ Resume training from checkpoints
- ✅ Flexible YAML configurations
- ✅ Advanced scripts for metrics analysis and visualization
- ✅ Docker support for deployment
- ✅ Automatic metrics export

## Installation

### 1. Clone/Download Required Files

Make sure you have the following files:
- `train_yolo.py` - Main training script with ablation studies
- `gen_dataset_v2.py` - Dataset creation and sampling tool
- `img_2_gray.py` - Image to grayscale conversion tool
- `requirements.txt` - Python dependencies
- `export_metrics.py` - Metrics export script
- `plot_curves.py` - Training curves plotting script
- `plot_grouped_metrics.py` - Grouped metrics visualization script
- `visualize_metrics.py` - Advanced visualizations script
- `Dockerfile` - Docker configuration for deployment
- `configs/` - Model configurations folder (YAML)
- `dataset_configs/` - Dataset configurations folder (YAML)
- `weights/` - Pre-trained weights folder

### 2. Docker Setup (Recommended)

The project is configured for Docker usage. See the [Docker Deployment](#docker-deployment) section for complete instructions.

**Quick Start:**
```bash
# Build the image
docker build -t cv2025cuda122img:v1 .

# Create and start container
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

### 3. Local Setup (Alternative)

If you prefer to install locally without Docker:

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# With Docker
docker exec -it cv2025cuda122containerV1GPU0 python train_yolo.py --help

# Local
python train_yolo.py --help
```

## Directory Structure

```
project/
├── train_yolo.py                 # Main training script with ablation
├── gen_dataset_v2.py             # Dataset creation tool
├── img_2_gray.py                 # Image to grayscale conversion
├── requirements.txt                  # Dependencies
├── export_metrics.py                # Metrics export
├── plot_curves.py                   # Training curves plotting
├── plot_grouped_metrics.py          # Grouped metrics visualization
├── visualize_metrics.py             # Advanced visualizations
├── Dockerfile                       # Docker configuration
├── configs/                         # Model configurations
│   ├── YOLOv10sLight.yaml          # YOLOv10 Light
│   ├── YOLOv12sLight.yaml          # YOLOv12 Light
│   ├── YOLOv12sLight050.yaml       # YOLOv12 Light 0.5
│   ├── YOLOv12sLightP3P4.yaml      # YOLOv12 Light P3P4
│   ├── YOLOv12sLightP3P4050.yaml   # YOLOv12 Light P3P4 0.5
│   ├── YOLOv12sLightP3P4AA.yaml    # YOLOv12 Light P3P4 AA
│   ├── YOLOv12sNormal.yaml         # YOLOv12 Normal
│   ├── YOLOv12sSM.yaml             # YOLOv12 SM
│   ├── YOLOv12sTurbo.yaml          # YOLOv12 Turbo
│   ├── YOLOv12sULT.yaml            # YOLOv12 Ultra
│   └── YOLOv12sUltraLight.yaml     # YOLOv12 Ultra Light
├── dataset_configs/                 # Dataset configurations
│   ├── d3_gray_dataset.yaml        # D3 Gray Dataset
│   └── DVM_1_Grey_dataset.yaml     # DVM-1 Grey Dataset
├── weights/                         # Pre-trained weights
│   ├── yolov10s.pt                 # YOLOv10s weights
│   ├── yolov12s.pt                 # YOLOv12s weights
│   └── best.pt                     # Other custom weights
└── results/                        # Training results (generated)
```

## Usage

### Basic Training

```bash
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### With Pre-trained Weights

```bash
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml --weights weights/yolov12s.pt
```

### Ablation Studies

Run ablation studies on different configurations:

```bash
# Test YOLOv12 Light vs Normal
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sNormal.yaml --dataset dataset_configs/d3_gray_dataset.yaml

# Compare P3P4 architectures
python train_yolo.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4050.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### Analysis and Visualization

After training, use the analysis scripts:

```bash
# Export metrics
python export_metrics.py --run_dir results/

# Create training curves plots
python plot_curves.py --results_dir results/

# Visualize grouped metrics for comparisons
python plot_grouped_metrics.py --experiments_dir results/

# Advanced visualizations
python visualize_metrics.py --data_path results/
```

## Main Parameters

| Parameter | Description | Required |
|-----------|-------------|-----------|
| `--config` | Model configuration YAML file | ✅ |
| `--dataset` | Dataset configuration YAML file | ✅ |
| `--weights` | Pre-trained weights (optional) | ❌ |

## Available Model Configurations

The project includes various configurations for ablation studies:

### YOLOv10
- **YOLOv10sLight**: YOLOv10 lightweight version

### YOLOv12 Base
- **YOLOv12sLight**: Standard lightweight version
- **YOLOv12sNormal**: Normal version
- **YOLOv12sTurbo**: Speed-optimized version
- **YOLOv12sUltraLight**: Ultra lightweight version

### YOLOv12 Experimental Variants
- **YOLOv12sLight050**: Light with 0.5 scaling
- **YOLOv12sLightP3P4**: Light with P3P4 head
- **YOLOv12sLightP3P4050**: Light P3P4 with 0.5 scaling
- **YOLOv12sSM**: Small Model variant
- **YOLOv12sULT**: Ultra variant

### Supported Datasets
- **d3_gray_dataset**: D3 grayscale dataset
- **DVM_1_Grey_dataset**: DVM-1 grayscale dataset

## Dataset Creation Tool

The project includes `gen_dataset_v2.py`, a powerful script for creating new datasets by combining and sampling from existing datasets.

### Features
- **Selective Sampling**: Choose specific number of images from multiple source datasets
- **Automatic Split**: Maintains train/validation/test split ratios
- **Label Verification**: Ensures image-label correspondence
- **Progress Tracking**: Real-time progress bars for dataset creation
- **Summary Reports**: Generates detailed statistics and contribution analysis
- **Error Handling**: Validates dataset availability before processing

### Usage

```bash
python gen_dataset_v2.py --dictionary "{dataset_dict}" --new-dataset "dataset_name" --split-ratio "[train,val,test]"
```

### Parameters

| Parameter | Description | Required | Format |
|-----------|-------------|-----------|---------|
| `--dictionary` | Dictionary specifying source datasets and image counts | ✅ | `'{"dataset1": count1, "dataset2": count2}'` |
| `--new-dataset` | Name of the new dataset to create | ✅ | String |
| `--split-ratio` | Train/validation/test split percentages | ✅ | `"[70,15,15]"` |

### Example: Creating DVM1 Dataset

Real command used to create the DVM1 dataset:

```bash
python gen_dataset_v2.py --dictionary "{'/datasets/S2_DETECTION_YOLO_png_640px': 1747, '/datasets/S2_DETECTION_YOLO_png_640px_rot': 3172, '/datasets/S2_DETECTION_YOLO_png_640px_rot_rot': 5963, '/datasets/S2_DETECTION_YOLO_png_640px_rot_rot_spe': 11919, '/datasets/S2_FC_png_640px': 1295, '/datasets/S2_FC_png_640px_rot': 2080, '/datasets/S2_FC_png_640px_rot_rot': 3596, '/datasets/S2_FC_png_640px_rot_rot_spe': 7164, '/datasets/SDAI_YOLO_png_640px': 621, '/datasets/SDAI_YOLO_png_640px_rot': 1230, '/datasets/SDAI_YOLO_png_640px_rot_rot': 2448, '/datasets/SDAI_YOLO_png_640px_rot_rot_spe': 4894}" --new-dataset DVM_1 --split-ratio "[70,15,15]"
```

### Output Structure

The script creates:
- **Dataset Directory**: `DVM_1/` with standard YOLO structure
  - `images/train/`, `images/val/`, `images/test/`
  - `labels/train/`, `labels/val/`, `labels/test/`
- **Summary Report**: `DVM_1.txt` with detailed statistics
- **Console Output**: Real-time progress and final summary

### Summary Report Contents
- Image counts per source dataset and split
- Contribution percentages to total dataset
- Label verification results
- Total dataset statistics

## Image Grayscale Conversion Tool

The project includes `img_2_gray.py`, a high-performance script for converting entire datasets to grayscale while preserving the complete directory structure.

### Features
- **Parallel Processing**: Multi-threaded conversion for optimal performance
- **Structure Preservation**: Maintains original directory hierarchy
- **Label Verification**: Ensures image-label correspondence after conversion
- **Progress Tracking**: Real-time progress bars for conversion process
- **File Type Detection**: Automatic image format recognition
- **Error Handling**: Robust file processing with detailed feedback

### Usage

```bash
python img_2_gray.py --path /path/to/dataset
```

### Parameters

| Parameter | Description | Required | Format |
|-----------|-------------|-----------|---------|
| `--path` | Path to the input dataset directory | ✅ | String (absolute or relative path) |

### Example: Converting DVM1 to Grayscale

Convert the DVM1 dataset to grayscale version:

```bash
python img_2_gray.py --path DVM_1
```

### Process Details

1. **Input Validation**: Verifies source directory exists
2. **Output Creation**: Creates new directory with `_Grey` suffix (e.g., `DVM_1_Grey`)
3. **Parallel Conversion**: Uses ThreadPoolExecutor for multi-core processing
4. **Image Processing**: Converts images to grayscale ('L' mode) using PIL
5. **Non-Image Files**: Copies other files (labels, configs) unchanged
6. **Label Verification**: Final check ensures all images have corresponding `.txt` labels

### Output Structure

For input dataset `DVM_1`, creates:
```
DVM_1_Grey/
├── images/
│   ├── train/          # Grayscale images
│   ├── val/            # Grayscale images  
│   └── test/           # Grayscale images
└── labels/
    ├── train/          # Unchanged label files
    ├── val/            # Unchanged label files
    └── test/           # Unchanged label files
```

### Supported Image Formats
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)
- GIF (`.gif`)

### Performance Features
- **Multi-threading**: Leverages multiple CPU cores
- **Memory Efficient**: Processes images one at a time
- **Progress Monitoring**: Real-time conversion progress
- **Automatic Verification**: Post-conversion label validation

## Pre-trained Weights

The `weights/` folder contains pre-trained weights to accelerate training:

### Weight Usage
- **Transfer Learning**: Load weights from pre-trained models for fine-tuning
- **Resume Training**: Resume training from saved checkpoints
- **Baseline Models**: Use already trained models as baselines for comparisons

### Usage Examples
```bash
# With YOLOv10 weights
python train_yolo.py --config configs/YOLOv10sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml --weights weights/yolov10s.pt

# With YOLOv12 weights
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml --weights weights/yolov12s.pt

# With custom checkpoint
python train_yolo.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/d3_gray_dataset.yaml --weights weights/best.pt
```

## Model Configuration (YAML)

Example YOLOv12 configuration:

```yaml
# YOLOv12 Configuration
nc: 1  # Number of classes
scales:
  s: [0.50, 0.50, 1024]  # small

# Backbone architecture
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  # ... other layers

# Head architecture  
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # ... other layers

# Training parameters
training:
  task: detect
  optimizer: SGD
  lr0: 0.01
  lrf: 0.0001
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3
  pretrained: False
  cache: False
  save_dir: "/workspace/results"
  project: "/workspace/results"
  device: 0
  epochs: 200
  batch: 64
  save: True
  verbose: True
  save_period: 10
```

## Dataset Configuration (YAML)

Example dataset configuration:

```yaml
# Dataset path configuration
path: /path/to/dataset
train: images/train
val: images/val  
test: images/test

# Class configuration
nc: 1  # Number of classes
names: ['ship']  # Class names
```

## Monitoring and Analysis

### TensorBoard
- Real-time metrics monitoring
- Loss curves, learning rate, mAP visualization
- Access via: `tensorboard --logdir=runs/tensorboard --host=0.0.0.0 --port=6006`

### Wandb Dashboard
- Advanced experiment management
- Comparisons between different configurations
- Hyperparameters and metrics tracking

### Analysis Scripts

1. **export_metrics.py**: Export metrics from TensorBoard/Wandb to CSV/JSON format
2. **plot_curves.py**: Generate training and validation curve plots
3. **plot_grouped_metrics.py**: Create comparative visualizations between experiments
4. **visualize_metrics.py**: Produce advanced visualizations and reports

### CO2 Emissions
- Automatic tracking with CodeCarbon
- Saved in `results/*/co2_production.txt`
- Detailed data in `results/*/emissions.csv`

## Results

Training results are saved in the directory specified in the YAML configuration file:
- `save_dir: "/workspace/results"`
- `project: "/workspace/results"`

```
results/
└── ModelName_DatasetName/
    ├── weights/
    │   ├── best.pt              # Best weights
    │   └── last.pt              # Last weights
    ├── args.yaml                # Training arguments
    ├── co2_production.txt       # CO2 emissions
    ├── emissions.csv            # CodeCarbon emission details
    ├── labels.jpg               # Labels visualization
    ├── train_batch*.jpg         # Training batches
    ├── runs/
    │   └── detect/
    │       └── ModelName_DatasetName_test/
    │           ├── confusion_matrix.png
    │           ├── results.png
    │           └── predictions/
    └── tensorboard/             # TensorBoard logs
        └── events.out.tfevents.*
```

## Ablation Studies Examples

### 1. Base Architecture Comparison

```bash
# YOLOv10 vs YOLOv12
python train_yolo.py --config configs/YOLOv10sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
```

### 2. Scaling Factors Study

```bash
# Compare scaling factors
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLight050.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### 3. Head Architecture Evaluation

```bash
# Standard vs P3P4 head
python train_yolo.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo.py --config configs/YOLOv12sLightP3P4AA.yaml --dataset dataset_configs/d3_gray_dataset.yaml
```

### 4. Transfer Learning Fine-tuning

```bash
# Fine-tuning from pre-trained weights
python train_yolo.py \
    --config configs/YOLOv12sTurbo.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml \
    --weights weights/yolov12s.pt
```

## Post-Training Analysis

After completing experiments, use the analysis scripts:

```bash
# 1. Export all metrics
python export_metrics.py --experiments_dir results/

# 2. Generate visual comparisons
python plot_grouped_metrics.py --experiments results/YOLOv12sLight_* results/YOLOv12sNormal_*

# 3. Create complete report
python visualize_metrics.py --results_dir results/ --output_dir analysis/
```

## Docker Deployment

### UnivPM Environment Setup

The project is configured for use with the UnivPM GPU server. Follow this procedure:

#### 1. Docker Image Build

```bash
docker build -t cv2025cuda122img:v1 .
```

#### 2. Container Creation

**Configure paths:**
- `/path/to/your/workspace`: Working directory containing code and where results will be saved
- `/path/to/your/datasets`: Directory containing datasets for training

To use different GPUs (GPU0 or GPU2) on the UnivPM server:

**Container for GPU0:**
```bash
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

**Container for GPU2:**
```bash
docker run -p6007:6006 -dit --shm-size=32g --gpus '"device=2"' \
    --name cv2025cuda122containerV1GPU2 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1
```

#### 3. Container Access

```bash
# For GPU0
docker exec -it cv2025cuda122containerV1GPU0 bash

# For GPU2
docker exec -it cv2025cuda122containerV1GPU2 bash
```

#### 4. Session Management with Tmux

Once inside the container, use `tmux` to manage persistent sessions:

**Create new session:**
```bash
tmux new -s yolo_train
```

**Exit session (keeping it active):**
```
CTRL+B + D
```

**Resume existing session:**
```bash
tmux attach -t yolo_train
```

**List active sessions:**
```bash
tmux ls
```

#### 5. Training Execution

Inside the tmux session:

```bash
# Training example
python train_yolo.py \
    --config configs/YOLOv12sTurbo.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml \
    --weights weights/yolov12s.pt
```

#### 6. TensorBoard Access

TensorBoard is accessible via browser:
- **GPU0**: `http://server-ip:6006`
- **GPU2**: `http://server-ip:6007`

#### Useful Tmux Commands
```bash
# List all sessions
tmux ls

# Create session with specific name
tmux new -s ablation_study_1

# Terminate session
tmux kill-session -t yolo_train

# Rename current session
tmux rename-session new_name
```

#### GPU Monitoring
```bash
# Check GPU usage
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

### Complete UnivPM Workflow

Example of complete workflow for ablation study:

```bash
# 1. Build image (once only)
docker build -t cv2025cuda122img:v1 .

# 2. Create GPU0 container
docker run -p6006:6006 -dit --shm-size=32g --gpus '"device=0"' \
    --name cv2025cuda122containerV1GPU0 \
    -v /path/to/your/workspace:/workspace \
    -v /path/to/your/datasets:/datasets \
    cv2025cuda122img:v1

# 3. Access container
docker exec -it cv2025cuda122containerV1GPU0 bash

# 4. Create tmux session
tmux new -s yolo_train

# 5. Start training
python train_yolo.py \
    --config configs/YOLOv12sLight.yaml \
    --dataset dataset_configs/DVM_1_Grey_dataset.yaml

# 6. Exit tmux (CTRL+B + D) to leave training running in background

# 7. In another session, monitor TensorBoard via browser:
# http://server-ip:6006

# 8. To resume monitoring:
tmux attach -t yolo_train
```

## Version Notes

- **Current version**: v7 (train_yolo.py)
- **Main features**: Ablation studies, TensorBoard integration, CO2 tracking
- **Compatibility**: YOLOv10/v12, Python 3.8+, CUDA 12.2+

## License

This project follows the licenses of the used libraries (Ultralytics YOLO, etc.).

## Author

**Valerio Marchetti**  
Project developed for the Computer Vision and Deep Learning course of Univesità Politecnica delle Marche (UnivPM), year 2025

---

**Created for the CVDL - Computer Vision and Deep Learning project**
