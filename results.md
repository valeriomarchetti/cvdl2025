# YOLO Model Configurations Comparison

**Project: YOLOv10/v12 Training & Analysis Suite**  
**Author: Valerio Marchetti**

This document provides a comprehensive comparison of all YOLO model configurations available in the `configs/` directory, designed for systematic ablation studies.

## Configuration Overview

The project includes **10 different configurations** across YOLOv10 and YOLOv12 architectures, each optimized for specific use cases and performance characteristics.

| Configuration | Version | Scale Factor | Output Heads | Primary Focus |
|---------------|---------|--------------|--------------|---------------|
| YOLOv10sLight | v10 | [0.33, 0.50] | P3 only | Lightweight v10 |
| YOLOv12sLight | v12 | [0.33, 0.50] | P3 only | Lightweight v12 |
| YOLOv12sLight050 | v12 | [0.33, 0.50] | P3 only | Light + 0.5 scaling |
| YOLOv12sLightP3P4 | v12 | [0.50, 0.50] | P3 + P4 | Multi-scale light |
| YOLOv12sLightP3P4050 | v12 | [0.50, 0.50] | P3 + P4 | P3P4 + 0.5 scaling |
| YOLOv12sLightP3P4AA | v12 | [0.50, 0.50] | P3 + P4 | P3P4 + attention |
| YOLOv12sNormal | v12 | [0.50, 0.50] | P3 + P4 + P5 | Standard normal |
| YOLOv12sSM | v12 | [0.50, 0.50] | P3 + P4 + P5 | Small model variant |
| YOLOv12sTurbo | v12 | [0.50, 0.50] | P3 + P4 + P5 | Speed optimized |
| YOLOv12sULT | v12 | [0.50, 0.50] | P3 + P4 + P5 | Ultra variant |
| YOLOv12sUltraLight | v12 | [0.33, 0.50] | P3 only | Ultra lightweight |

## Detailed Architecture Comparison

### 1. YOLOv10sLight
**Purpose**: Baseline YOLOv10 lightweight implementation
- **Backbone**: Traditional YOLOv10 with C2f, SCDown, C2fCIB modules
- **Head**: Single P3 output with v10Detect
- **Special Features**: PSA attention, SPPF pooling
- **Scale**: [0.33, 0.50, 1024]
- **Detection**: Single-scale P3 only
- **Use Case**: Reference implementation for v10 vs v12 comparison

### 2. YOLOv12sLight  
**Purpose**: Lightweight YOLOv12 with minimal complexity
- **Backbone**: Simplified with C3k2 and A2C2f modules
- **Head**: Single P3 output 
- **Scale**: [0.33, 0.50, 1024]
- **Detection**: P3 only (fastest inference)
- **Use Case**: Speed-critical applications, edge deployment

### 3. YOLOv12sLight050
**Purpose**: YOLOv12 Light with modified scaling
- **Backbone**: Same as YOLOv12sLight
- **Head**: Single P3 output
- **Scale**: [0.33, 0.50, 1024] with 0.5 scaling modifications
- **Detection**: P3 only
- **Use Case**: Testing scaling factor impact on performance

### 4. YOLOv12sLightP3P4
**Purpose**: Multi-scale detection with P3+P4 heads
- **Backbone**: Standard YOLOv12 backbone
- **Head**: Dual-scale P3 + P4 detection
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: P3 (small objects) + P4 (medium objects)
- **Use Case**: Balanced speed vs accuracy for varied object sizes

### 5. YOLOv12sLightP3P4050
**Purpose**: P3P4 variant with 0.5 scaling modifications
- **Backbone**: YOLOv12 with scaling adjustments
- **Head**: P3 + P4 outputs
- **Scale**: [0.50, 0.50, 1024] + 0.5 scaling
- **Detection**: Dual-scale P3 + P4
- **Use Case**: Scaling factor ablation on multi-scale detection

### 6. YOLOv12sLightP3P4AA
**Purpose**: P3P4 with enhanced attention mechanisms
- **Backbone**: YOLOv12 + additional attention layers
- **Head**: P3 + P4 with attention augmentation
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: Attention-enhanced P3 + P4
- **Use Case**: Attention mechanism ablation study

### 7. YOLOv12sNormal
**Purpose**: Standard YOLOv12 implementation
- **Backbone**: Full YOLOv12 backbone
- **Head**: Complete P3 + P4 + P5 detection
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: Full multi-scale (P3/P4/P5)
- **Use Case**: Baseline for performance comparison

### 8. YOLOv12sSM (Small Model)
**Purpose**: Compact variant with grouped convolutions
- **Backbone**: YOLOv12 with Conv groups (2, 4)
- **Head**: P3 + P4 + P5 with A2C2f
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: Full multi-scale
- **Special Features**: Grouped convolutions for efficiency
- **Use Case**: Resource-constrained environments

### 9. YOLOv12sTurbo
**Purpose**: Speed-optimized with area attention
- **Backbone**: YOLOv12 + grouped convolutions + A2C2f
- **Head**: P3 + P4 + P5 with optimized attention
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: Full multi-scale
- **Special Features**: Area attention (A2C2f) for speed
- **Use Case**: Real-time applications requiring accuracy

### 10. YOLOv12sULT (Ultra)
**Purpose**: Advanced variant with enhanced features
- **Backbone**: YOLOv12 with ultra optimizations
- **Head**: P3 + P4 + P5 
- **Scale**: [0.50, 0.50, 1024]
- **Detection**: Full multi-scale
- **Use Case**: Maximum performance scenarios

### 11. YOLOv12sUltraLight
**Purpose**: Minimal architecture for extreme speed
- **Backbone**: Heavily simplified (4 layers only)
- **Head**: Single P3 output
- **Scale**: [0.33, 0.50, 1024]
- **Detection**: P3 only
- **Use Case**: Extreme edge deployment, IoT devices

## Architecture Components Analysis

### Backbone Modules
| Module | Purpose | Configs Using |
|--------|---------|---------------|
| **Conv** | Basic convolution | All configurations |
| **C2f** | YOLOv8-style bottleneck | YOLOv10sLight |
| **C3k2** | YOLOv12 bottleneck | All YOLOv12 variants |
| **A2C2f** | Area attention C2f | Most YOLOv12 variants |
| **SCDown** | Spatial channel downsampling | YOLOv10sLight |
| **C2fCIB** | Channel interaction block | YOLOv10sLight |
| **PSA** | Position-sensitive attention | YOLOv10sLight |
| **SPPF** | Spatial pyramid pooling | YOLOv10sLight |

### Head Detection Patterns
| Pattern | Configurations | Object Size Coverage |
|---------|----------------|---------------------|
| **P3 Only** | Light, UltraLight, Light050 | Small objects only |
| **P3 + P4** | LightP3P4, LightP3P4050, LightP3P4AA | Small + Medium objects |
| **P3 + P4 + P5** | Normal, SM, Turbo, ULT | All object sizes |

### Scale Factor Impact
| Scale Factor | Configurations | Model Size | Speed | Accuracy |
|-------------|----------------|------------|-------|----------|
| **[0.33, 0.50]** | Light, UltraLight | Smallest | Fastest | Basic |
| **[0.50, 0.50]** | Normal, Turbo, SM, ULT, P3P4 variants | Medium | Balanced | Higher |

## Training Configuration Comparison

All configurations share common training parameters:

```yaml
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
  workers: 0
  amp: True  # Automatic Mixed Precision
```

## Recommended Ablation Studies

### 1. Architecture Version Comparison
```bash
# YOLOv10 vs YOLOv12 baseline
python train_yolo_v7.py --config configs/YOLOv10sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
python train_yolo_v7.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml
```

### 2. Scale Factor Impact
```bash
# Scale factor ablation
python train_yolo_v7.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
python train_yolo_v7.py --config configs/YOLOv12sLight050.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml
```

### 3. Multi-Scale Detection Study
```bash
# Single vs multi-scale heads
python train_yolo_v7.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/d3_gray_dataset.yaml  # P3 only
python train_yolo_v7.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/d3_gray_dataset.yaml  # P3+P4
python train_yolo_v7.py --config configs/YOLOv12sNormal.yaml --dataset dataset_configs/d3_gray_dataset.yaml  # P3+P4+P5
```

### 4. Attention Mechanism Analysis
```bash
# Attention impact study
python train_yolo_v7.py --config configs/YOLOv12sLightP3P4.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml  # Standard
python train_yolo_v7.py --config configs/YOLOv12sLightP3P4AA.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml  # +Attention
```

### 5. Optimization Variants Comparison
```bash
# Performance optimization study
python train_yolo_v7.py --config configs/YOLOv12sNormal.yaml --dataset dataset_configs/d3_gray_dataset.yaml    # Baseline
python train_yolo_v7.py --config configs/YOLOv12sSM.yaml --dataset dataset_configs/d3_gray_dataset.yaml        # Small Model
python train_yolo_v7.py --config configs/YOLOv12sTurbo.yaml --dataset dataset_configs/d3_gray_dataset.yaml     # Turbo
python train_yolo_v7.py --config configs/YOLOv12sULT.yaml --dataset dataset_configs/d3_gray_dataset.yaml       # Ultra
```

### 6. Extreme Lightweight Study
```bash
# Lightweight variants comparison
python train_yolo_v7.py --config configs/YOLOv12sLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml       # Light
python train_yolo_v7.py --config configs/YOLOv12sUltraLight.yaml --dataset dataset_configs/DVM_1_Grey_dataset.yaml  # Ultra Light
```

## Expected Performance Characteristics

### Speed Ranking (Fastest → Slowest)
1. **YOLOv12sUltraLight** - Minimal architecture
2. **YOLOv12sLight** - Single P3 head
3. **YOLOv12sLight050** - Light with scaling
4. **YOLOv12sTurbo** - Speed optimized
5. **YOLOv12sSM** - Small model with groups
6. **YOLOv12sLightP3P4** - Dual head
7. **YOLOv12sLightP3P4050** - Dual head + scaling
8. **YOLOv12sLightP3P4AA** - Dual head + attention
9. **YOLOv12sNormal** - Full three heads
10. **YOLOv12sULT** - Ultra features
11. **YOLOv10sLight** - Traditional architecture

### Accuracy Ranking (Expected High → Low)
1. **YOLOv12sULT** - Advanced features
2. **YOLOv12sNormal** - Full multi-scale
3. **YOLOv12sTurbo** - Optimized attention
4. **YOLOv12sLightP3P4AA** - Attention enhanced
5. **YOLOv12sSM** - Efficient architecture
6. **YOLOv12sLightP3P4** - Dual scale
7. **YOLOv12sLightP3P4050** - Dual scale + scaling
8. **YOLOv10sLight** - Traditional approach
9. **YOLOv12sLight** - Single scale
10. **YOLOv12sLight050** - Single scale + scaling
11. **YOLOv12sUltraLight** - Minimal features

### Memory Usage Ranking (Lowest → Highest)
1. **YOLOv12sUltraLight** - 4 backbone layers
2. **YOLOv12sLight** - Simple single head
3. **YOLOv12sLight050** - Light with modifications
4. **YOLOv12sSM** - Grouped convolutions
5. **YOLOv12sTurbo** - Optimized but full
6. **YOLOv12sLightP3P4** - Dual head baseline
7. **YOLOv12sLightP3P4050** - Dual head + features
8. **YOLOv12sLightP3P4AA** - Dual head + attention
9. **YOLOv10sLight** - Traditional modules
10. **YOLOv12sNormal** - Full architecture
11. **YOLOv12sULT** - Maximum features

## Analysis Recommendations

### For Speed-Critical Applications
- Start with **YOLOv12sUltraLight** or **YOLOv12sLight**
- Compare against **YOLOv12sTurbo** for speed vs accuracy trade-off

### For Accuracy-Critical Applications  
- Begin with **YOLOv12sNormal** as baseline
- Test **YOLOv12sULT** for maximum performance
- Try **YOLOv12sLightP3P4AA** for attention benefits

### For Resource-Constrained Environments
- **YOLOv12sSM** with grouped convolutions
- **YOLOv12sUltraLight** for extreme constraints
- Compare memory usage vs accuracy trade-offs

### For Balanced Performance
- **YOLOv12sTurbo** - good speed/accuracy balance
- **YOLOv12sLightP3P4** - dual-scale detection
- **YOLOv12sNormal** - reliable baseline

## Systematic Testing Protocol

1. **Single Configuration Testing**: Validate each config individually
2. **Pairwise Comparisons**: Direct A/B testing between similar configs
3. **Progressive Studies**: Light → Normal → Ultra progression
4. **Cross-Dataset Validation**: Test on both d3_gray and DVM_1_Grey
5. **Statistical Analysis**: Multiple runs for significance testing

---

**Generated for CVDL 2025 - Computer Vision and Deep Learning Project**  
**Valerio Marchetti - Ablation Studies Documentation**
