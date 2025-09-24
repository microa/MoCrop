# MoCrop: Training Free Motion Guided Cropping for Efficient Video Action Recognition

A PyTorch implementation of MoCrop, a motion-aware cropping technique for efficient video action recognition that leverages motion vectors from compressed video streams.

## Overview

MoCrop is a plug-and-play module that can be integrated with various backbone architectures (ResNet, MobileNet, EfficientNet, Swin Transformer) to improve both accuracy and efficiency in video action recognition. The method uses motion vectors extracted from compressed video streams to intelligently crop video frames, focusing on motion-rich regions while reducing computational costs.

## Key Features

- **Motion-Aware Cropping**: Uses motion vectors to identify and crop salient regions in video frames
- **Multiple Backbone Support**: Compatible with ResNet-50/152, MobileNet-V3-Large, EfficientNet-B1, and Swin-B
- **Dual Advantages**: Provides both accuracy improvement and computational efficiency
- **Easy Integration**: Simple integration with existing training and testing pipelines
- **Flexible Settings**: Supports both efficiency-focused and attention-focused cropping modes

## Architecture

The MoCrop framework consists of:
- **Motion Vector Processing**: Extracts and processes motion vectors from compressed videos
- **Adaptive Cropping**: Uses motion density maps to determine optimal crop regions
- **Backbone Integration**: Seamlessly integrates with various CNN and Transformer architectures

## Requirements

- Python 3.8
- PyTorch 1.9.0
- torchvision 0.10.0
- opencv-python 4.5.0
- numpy 1.21.0
- tqdm 4.64.0
- fvcore 0.1.5
- matplotlib 3.5.0
- Pillow 8.3.0
- scipy 1.7.0
- scikit-learn 1.0.0

## Installation

Clone the repository:
```bash
git clone https://github.com/microa/MoCrop.git
cd MoCrop
```

Install dependencies:
```bash
conda create -n mocrop python=3.8
conda activate mocrop
pip install torch torchvision opencv-python numpy tqdm fvcore matplotlib Pillow scipy scikit-learn
```

## Dataset Preparation

1. Download UCF-101 dataset and extract to your data directory
2. Convert videos to MP4 format (H.264 codec)
3. Extract motion vectors and save as .npy files
4. Prepare dataset list files in the format: video.avi label

## Usage

### Training

Train models with MoCrop:
```bash
# ResNet-50 with MoCrop
python scripts/train.py --arch resnet50 --train-mode mocrop --epochs 100 --batch-size 32

# EfficientNet-B1 with MoCrop  
python scripts/train.py --arch efficientnet_b1 --train-mode mocrop --epochs 100 --batch-size 32

# MobileNet-V3-Large with MoCrop
python scripts/train.py --arch mobilenet_v3_large --train-mode mocrop --epochs 100 --batch-size 32

# Swin-B with MoCrop
python scripts/train.py --arch swin_b --train-mode mocrop --epochs 100 --batch-size 32
```

Train models without MoCrop (baseline):
```bash
python scripts/train.py --arch resnet50 --train-mode normal --epochs 100 --batch-size 32
```

### Testing

Test trained models:
```bash
# Test with MoCrop
python scripts/test.py --arch resnet50 --test-mode mocrop --model-path path/to/model.pth

# Test without MoCrop (baseline)
python scripts/test.py --arch resnet50 --test-mode normal --model-path path/to/model.pth
```

## Experimental Results

Results on UCF-101 Split 1:

| Backbone | Variant | Top-1 Acc. (%) | GFLOPs | Δ Acc. (%) | Δ FLOPs (%) |
|----------|---------|----------------|--------|------------|-------------|
| ResNet-50 | Baseline (224px) | 80.1 | 4.11 | - | - |
| ResNet-50 | + MoCrop (Efficiency, 192px) | 82.5 | 3.02 | +2.4 | -26.5 |
| ResNet-50 | + MoCrop (Attention, 224px) | 83.6 | 4.11 | +3.5 | 0.0 |
| MobileNet-V3 | Baseline (224px) | 78.3 | 0.22 | - | - |
| MobileNet-V3 | + MoCrop (Efficiency, 192px) | 79.3 | 0.17 | +1.0 | -22.7 |
| MobileNet-V3 | + MoCrop (Attention, 224px) | 80.2 | 0.22 | +1.9 | 0.0 |
| EfficientNet-B1 | Baseline (224px) | 82.1 | 0.59 | - | - |
| EfficientNet-B1 | + MoCrop (Efficiency, 192px) | 83.6 | 0.43 | +1.5 | -27.1 |
| EfficientNet-B1 | + MoCrop (Attention, 224px) | 84.5 | 0.59 | +2.4 | 0.0 |
| Swin-B | Baseline (224px) | 87.3 | 15.5 | - | - |
| Swin-B | + MoCrop (Efficiency, 192px) | 87.2 | 12.6 | -0.1 | -18.7 |
| Swin-B | + MoCrop (Attention, 224px) | 87.5 | 15.5 | +0.2 | 0.0 |

## Experimental Evidence

The experimental results shown above are fully reproducible. Detailed execution logs and accuracy outputs for all models under both normal and MoCrop testing conditions are available in notebooks/test_results.ipynb. This notebook contains:

- Complete command execution logs
- Model loading progress and details
- Batch-by-batch processing information
- Final accuracy calculations and counts
- Performance metrics for all tested configurations

## Project Structure

```
MoCrop/
 src/                          # Core source code
    mocrop_dataset.py         # MoCrop dataset class (core algorithm)
    models.py                 # Model definitions
    transforms.py             # Data transformations
 scripts/                      # Training and testing scripts
    train.py                  # Training script
    test.py                   # Testing script
    calculate_flops.py        # FLOPs calculation script
 notebooks/                    # Experimental logs
    test_results.ipynb       # Experimental results (corresponds to Table 1)
 README.md                     # This file
 requirements.txt              # Python dependencies
 setup.py                      # Installation script
 environment.yml               # Conda environment configuration
 LICENSE                       # MIT License
```

## Important Notes

- **Training Mode**: Use --train-mode normal for baseline training, --train-mode mocrop for MoCrop training
- **Motion Vectors**: MoCrop mode requires .npy files containing motion vectors
- **Parameter Consistency**: Ensure --mv-h and --mv-w parameters match the motion vector file dimensions
- **Fallback Behavior**: If motion vector files are missing, the system falls back to using original frames

## Motion Vector Visualization

The framework includes tools for visualizing motion vectors and understanding the cropping decisions made by MoCrop. Motion density maps show the concentration of motion activity, which guides the adaptive cropping process.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{huang2025mocrop,
  title={MoCrop: Training Free Motion Guided Cropping for Efficient Video Action Recognition},
  author={Huang, Binhua and Yao, Wendong and Chen, Shaowu and Wang, Guoxin and Wang, Qingyuan and Dev, Soumyabrata},
  journal={arXiv preprint arXiv:2509.18473},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCF-101 dataset creators
- PyTorch and the open-source community
- Various backbone architecture implementations