# HiVT-AV2: Enhanced Hierarchical Vector Transformer for Motion Forecasting

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An enhanced implementation of the Hierarchical Vector Transformer (HiVT) model for autonomous vehicle motion forecasting, adapted to work with the Argoverse 2 Motion Forecasting dataset. This implementation extends the [original HiVT model](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf) by incorporating additional vehicle state information including velocity and heading angle while maintaining rotation and translation invariance.

## Key Features

- Supports Argoverse 2 Motion Forecasting dataset
- Enhanced feature representation with position, velocity, and heading angle
- Maintains rotation and translation invariance
- Hierarchical architecture with local and global interactions
- Multi-modal trajectory prediction
- Multiple evaluation metrics (minADE, minFDE, MR)

## Model Architecture

The model consists of three main components:

1. **Local Encoder**: Processes individual agent histories
2. **Global Interactor**: Models interactions between agents
3. **Decoder**: Generates multi-modal trajectory predictions

## Installation

### Dependencies

```bash
# PyTorch ecosystem
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pytorch.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric pytorch_lightning==1.9.0 pandas numpy==1.25.0

# Argoverse 2 API
pip install "av2@git+https://github.com/argoverse/av2-api.git@main"
```

## Dataset

### Download and Preparation

1. Download the Argoverse 2 Motion Forecasting dataset from the [official website](https://www.argoverse.org/av2.html)
2. Extract the dataset to your preferred location

### Required Directory Structure

The dataset must be organized according to the following structure for proper functionality:

```
dataset/
├── train/
│   └── scenarios/
│       └── <scenario_id>/
│           ├── <scenario_id>.parquet    # Vehicle trajectory data
│           └── log_map_archive_*.json    # HD map data
├── val/
│   └── scenarios/
│       └── <scenario_id>/
│           ├── <scenario_id>.parquet
│           └── log_map_archive_*.json
```

**File descriptions:**
- `.parquet` files: Contains timestamped vehicle trajectory data including positions, velocities, and headings
- `.json` files: Contains HD map information including lane geometries, traffic controls, and intersections

## Usage

### Training

Train the model with a 64-dimensional embedding:

```bash
python train_v2.py --root /path/to/dataset/ --embed_dim 64
```

or with a 128-dimensional embedding:

```bash
python train_v2.py --root /path/to/dataset/ --embed_dim 128
```

### Evaluation

Evaluate a trained model:

```bash
python eval_v2.py --root /path/to/dataset/ --ckpt_path /path/to/checkpoint/
```

## Model Parameters

The main configurable parameters include:

- `--historical_steps`: Number of historical timesteps (default: 50)
- `--future_steps`: Number of future timesteps to predict (default: 60)
- `--num_modes`: Number of prediction modes (default: 6)
- `--rotate`: Whether to use rotation invariance (default: True)
- `--node_dim`: Input dimension for node features (default: 5 for [x, y, vx, vy, heading])
- `--embed_dim`: Embedding dimension (required parameter)
- `--num_heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout rate (default: 0.1)
- `--num_temporal_layers`: Number of temporal encoder layers (default: 4)
- `--num_global_layers`: Number of global interaction layers (default: 3)
- `--local_radius`: Radius for local interactions (default: 50)
- `--lr`: Learning rate (default: 5e-4)
- `--weight_decay`: Weight decay (default: 1e-4)

## Project Structure

```
├── assets/                  # Visualization and documentation assets
├── checkpoints/             # Trained model checkpoints
├── datamodules/             # PyTorch Lightning datamodules
├── datasets/                # Dataset implementations
├── losses/                  # Loss function implementations
├── metrics/                 # Evaluation metrics
├── models/                  # Model architecture implementations
├── eval_v2.py               # Evaluation script for Argoverse 2
├── eval.py                  # Original evaluation script for Argoverse 1
├── train_v2.py              # Training script for Argoverse 2
├── train.py                 # Original training script for Argoverse 1
└── utils.py                 # Utility functions
```

## Citation

If you use this implementation in your work, please cite the original HiVT paper:

```bibtex
@inproceedings{zhou2022hivt,
  title={HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction},
  author={Zhou, Zikang and Luo, Jianping and Bai, Shangjie and Wang, Haosheng and Zhao, Hongsheng and Wang, Ying and Yang, Li and Chai, Wei and Li, Hao and Huang, Hongming and Liu, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2312--2321},
  year={2022}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.