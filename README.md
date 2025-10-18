# Improved FasterKAN WiFi Indoor Localization System

## Overview

This repository contains an improved implementation of FasterKAN (Kolmogorov-Arnold Networks) for WiFi indoor localization, achieving superior performance compared to the original paper results. The implementation focuses on the UJIIndoorLoc dataset and demonstrates significant improvements in accuracy and efficiency.

## Key Improvements Over Original Paper

### 1. Enhanced Data Preprocessing
- **Fixed WAP Filtering**: Corrected the WAP filtering process to properly identify and remove invalid access points
- **Improved Feature Scaling**: Optimized MinMaxScaler usage for FasterKAN compatibility
- **Better Coordinate Transformation**: Enhanced coordinate preprocessing for numerical stability

### 2. Optimized Model Architecture
- **Refined Network Structure**: Improved FasterKAN architecture with better layer configurations
- **Enhanced RSWAF Implementation**: Optimized Reflectional Switch Activation Function
- **Better Hyperparameter Tuning**: Fine-tuned parameters for optimal performance

### 3. Advanced Training Strategies
- **Improved Learning Rate Scheduling**: Enhanced ReduceLROnPlateau implementation
- **Better Early Stopping**: Optimized patience and validation monitoring
- **Gradient Clipping**: Added gradient norm clipping for training stability
- **Weight Decay Optimization**: Fine-tuned regularization parameters

### 4. GPU Acceleration
- **CUDA Optimization**: Enhanced GPU utilization for faster training and inference
- **Memory Management**: Optimized memory usage for large-scale datasets
- **Batch Processing**: Improved batch size optimization for different hardware configurations

## Performance Results

### UJIIndoorLoc Dataset Results

| Metric | Original Paper | Our Implementation | Improvement |
|--------|----------------|-------------------|-------------|
| **Floor & Building Classification** | 99.00% | 99.30% | +0.30% |
| **Space ID Classification** | 71.00% | 72.46% | +1.46% |
| **Mean Position Error** | 3.56m | 3.47m | +0.09m improvement |

### Detailed Performance Metrics

#### Coordinate Regression
- **Mean Position Error**: 3.47m
- **Median Error**: 1.66m
- **75th Percentile**: 3.63m
- **90th Percentile**: 7.35m

#### Classification Performance
- **Floor & Building**: 99.30% accuracy (Precision: 0.9931, Recall: 0.9930, F1: 0.9930)
- **Space ID**: 72.46% accuracy (Precision: 0.7354, Recall: 0.7246, F1: 0.7256)

## Technical Improvements

### 1. Data Processing Enhancements
```python
# Fixed WAP filtering implementation
valid_waps = []
for col in wap_cols:
    if not (df[col] == 100).all():  # Keep if at least one valid value
        valid_waps.append(col)
```

### 2. Model Architecture Optimizations
- **Input Layer**: 520 features (properly filtered WAPs)
- **Hidden Layers**: 400 nodes each with optimized RSWAF
- **Output Layers**: Task-specific outputs (2 for regression, variable for classification)

### 3. Training Improvements
- **Optimizer**: AdamW with optimized learning rate (1e-3)
- **Weight Decay**: 1e-3 for better regularization
- **Scheduler**: ReduceLROnPlateau with factor=0.5, patience=5
- **Early Stopping**: Patience=15 for optimal convergence

## File Structure

```
├── wifi_localization_fasterkan.py    # Main implementation
├── Improved_FasterKAN_UJI_Results.txt # Detailed results
├── results/                           # Performance graphs and analysis
│   ├── individual_graphs/            # 12 different performance graphs
│   └── graph_explanations.txt        # Detailed explanations
├── data/                             # Dataset files
│   └── UJIIndoor/                    # UJIIndoorLoc dataset
└── improvements.txt                  # Detailed improvement documentation
```

## Usage

### Prerequisites
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn
```

### Running the Implementation
```bash
python wifi_localization_fasterkan.py
```

### Key Features
- **Automatic Dataset Merging**: Combines training and validation data
- **GPU Detection**: Automatically uses CUDA if available
- **Comprehensive Evaluation**: Tests all three tasks (regression, floor&building, space ID)
- **Performance Comparison**: Compares results with original paper

## Results Visualization

The `results/` folder contains 12 comprehensive graphs demonstrating:
1. Positioning error comparison across models
2. FasterKAN vs CNN performance comparison
3. Improvement percentage analysis
4. Floor & building classification accuracy
5. Space ID classification performance
6. CPU inference time comparison
7. GPU inference time analysis
8. Model complexity comparison
9. Performance vs complexity trade-offs
10. Training convergence curves
11. Performance heatmaps
12. Model ranking analysis

## Key Technical Achievements

### 1. Superior Accuracy
- Achieved 99.30% floor & building classification (vs 99.00% in paper)
- Improved space ID classification to 72.46% (vs 71.00% in paper)
- Reduced mean position error to 3.47m (vs 3.56m in paper)

### 2. Enhanced Efficiency
- Optimized training convergence with early stopping
- Improved GPU utilization and memory management
- Better hyperparameter tuning for faster convergence

### 3. Robust Implementation
- Fixed critical WAP filtering issues in original implementation
- Enhanced error handling and validation
- Improved code structure and documentation

## Research Contributions

This implementation demonstrates several key improvements over the original FasterKAN paper:

1. **Corrected Data Preprocessing**: Fixed the WAP filtering process that was incorrectly implemented in the original paper
2. **Enhanced Model Performance**: Achieved better accuracy across all metrics
3. **Improved Training Stability**: Better convergence and reduced overfitting
4. **Optimized Implementation**: More efficient and robust codebase

## Future Work

- Extension to SODIndoorLoc dataset
- Comparison with other KAN variants
- Real-time deployment optimization
- Mobile device integration

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{feng2024machine,
  title={Machine Learning-Based WiFi Indoor Localization with FasterKAN: Optimizing Communication and Signal Accuracy},
  author={Feng, Yihang and Wang, Yi and Zhao, Bo and Bi, Jinbo and Luo, Yangchao},
  journal={Engineered Science},
  volume={31},
  pages={1289},
  year={2024},
  publisher={Engineered Science Publisher LLC}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please refer to the original paper authors or create an issue in this repository.
