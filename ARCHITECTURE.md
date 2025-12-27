# Model Architecture Documentation

## Overview

This document describes the architecture of the Multimodal Property Valuation Model that combines tabular data with satellite imagery features.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  TABULAR DATA    │              │ SATELLITE IMAGES  │         │
│  │                  │              │                  │         │
│  │ • bedrooms       │              │  Property Images │         │
│  │ • bathrooms      │              │  (lat, long)     │         │
│  │ • sqft_living    │              │                  │         │
│  │ • lat, long      │              │  [256x256 PNG]   │         │
│  │ • grade          │              │                  │         │
│  │ • waterfront     │              └────────┬─────────┘         │
│  │ • view           │                       │                   │
│  │ • ...            │                       │                   │
│  └────────┬─────────┘                       │                   │
│           │                                 │                   │
└───────────┼─────────────────────────────────┼───────────────────┘
            │                                 │
            │                                 │
┌───────────▼─────────────────────────────────▼───────────────────┐
│                    FEATURE ENGINEERING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  TABULAR FEATURES│              │  VISUAL FEATURES │         │
│  │                  │              │                  │         │
│  │ • Log transform  │              │  ResNet18 CNN    │         │
│  │ • Geospatial:    │              │  Feature         │         │
│  │   - dist_to_hub  │              │  Extractor       │         │
│  │   - dist_center  │              │                  │         │
│  │ • Temporal:      │              │  Output:         │         │
│  │   - house_age    │              │  visual_score    │         │
│  │ • Ratios:        │              │  (scalar)        │         │
│  │   - price/sqft   │              │                  │         │
│  │   - living_ratio │              │                  │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
│           │                                 │                   │
└───────────┼─────────────────────────────────┼───────────────────┘
            │                                 │
            └─────────────┬───────────────────┘
                          │
                          │ FUSION
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    FEATURE CONCATENATION                         │
│                                                                   │
│  [Tabular Features (14) + Visual Score (1)] = 15 Features       │
│                                                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    ENSEMBLE MODEL                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              VOTING REGRESSOR                             │   │
│  │                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │   Random    │  │   XGBoost    │  │   CatBoost   │   │   │
│  │  │   Forest    │  │              │  │              │   │   │
│  │  │   (20%)     │  │   (40%)      │  │   (40%)      │   │   │
│  │  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘   │   │
│  │         │                │                 │           │   │
│  │         └────────────────┼─────────────────┘           │   │
│  │                          │                             │   │
│  │                  Weighted Average                      │   │
│  └──────────────────────────┬─────────────────────────────┘   │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              │
                    ┌─────────▼─────────┐
                    │   PREDICTIONS     │
                    │                   │
                    │  log_price        │
                    │  → expm1()        │
                    │  → USD Price      │
                    └───────────────────┘
```

## Component Details

### 1. Input Data

**Tabular Data:**
- Property features: bedrooms, bathrooms, square footage, etc.
- Location: latitude, longitude
- Quality indicators: grade, condition, view, waterfront
- Neighborhood: sqft_living15, sqft_lot15

**Satellite Images:**
- High-resolution satellite imagery (256x256 pixels)
- Downloaded using ESRI World Imagery tile service
- Zoom level 19 for detailed view

### 2. Feature Engineering

**Tabular Features:**
- **Log Transformations**: Applied to skewed features (price, sqft_living, etc.)
- **Geospatial Features**:
  - `dist_to_hub`: Distance to luxury property hub (top 5% properties)
  - `dist_from_center`: Distance from dataset geographic center
- **Temporal Features**:
  - `house_age`: Years since construction (2015 - yr_built)
  - `is_renovated`: Binary indicator for renovation
- **Ratio Features**:
  - `price_per_sqft`: Price per square foot of living space
  - `living_ratio`: Living space to lot size ratio
  - `bed_bath_ratio`: Bedroom to bathroom ratio

**Visual Features:**
- **CNN Feature Extractor**: Pre-trained ResNet18 (ImageNet weights)
- **Process**:
  1. Image preprocessing (resize, normalize)
  2. Feature extraction (remove final classification layer)
  3. Global average pooling → `visual_score` (scalar value)
- **Output**: Single scalar `visual_score` representing visual quality/environmental context

### 3. Feature Fusion

**Concatenation Strategy:**
- Simple concatenation of tabular features (14) + visual_score (1) = 15 features
- No early or late fusion - all features treated equally by ensemble model

**Alternative Approaches Considered:**
- Early fusion: Direct CNN feature vector (512-dim) concatenation (too high-dimensional)
- Late fusion: Separate models for tabular and visual, then combine predictions
- **Chosen**: Scalar visual_score for simplicity and interpretability

### 4. Ensemble Model

**Voting Regressor:**
- Combines predictions from 3 base models with weighted averaging
- **Random Forest** (20% weight):
  - 500 trees, max_depth=15
  - Captures non-linear relationships and interactions
- **XGBoost** (40% weight):
  - 1200 estimators, learning_rate=0.03, max_depth=8
  - Gradient boosting for complex patterns
- **CatBoost** (40% weight):
  - 1200 iterations, learning_rate=0.03, depth=8
  - Handles categorical features effectively

**Training:**
- Target: `log_price` (log-transformed price for better convergence)
- Final predictions: `expm1(predicted_log_price)` to convert back to USD

## Data Flow

1. **Data Loading**: Load CSV and verify image paths
2. **Preprocessing**: Remove outliers (top 1% prices), handle missing values
3. **Feature Engineering**: Create geospatial, temporal, and ratio features
4. **Visual Extraction**: Process all images through CNN to get visual_score
5. **Train-Test Split**: 80/20 split (ensuring no data leakage)
6. **Model Training**: Fit ensemble model on training data
7. **Prediction**: Generate predictions on test set
8. **Evaluation**: Calculate R², MAE, RMSE metrics

## Key Design Decisions

1. **ResNet18 as Feature Extractor**: 
   - Pre-trained on ImageNet, captures general visual patterns
   - Lightweight compared to ResNet50/101
   - Sufficient for satellite imagery feature extraction

2. **Scalar Visual Score**:
   - Reduces dimensionality from 512-dim feature vector to 1 scalar
   - Easier to interpret and integrate with tabular features
   - Prevents overfitting with limited training data

3. **Ensemble Approach**:
   - Voting regressor combines strengths of different algorithms
   - Weights favor gradient boosting (XGBoost, CatBoost) for accuracy
   - Random Forest adds diversity and robustness

4. **Log Transformation**:
   - Price distribution is highly skewed
   - Log transform normalizes distribution and improves model convergence
   - Standard practice in real estate valuation

## Performance Characteristics

- **Training Time**: ~10-15 minutes (depending on hardware)
- **Inference Time**: ~1-2 seconds per property (including image processing)
- **Model Size**: ~500MB (ensemble of 3 models)
- **Memory Usage**: ~4-6GB during training

## Future Improvements

1. **Advanced Fusion**: Try attention mechanisms or learned fusion layers
2. **Larger CNN**: Experiment with ResNet50 or EfficientNet for richer visual features
3. **Multi-scale Images**: Use images at different zoom levels
4. **Temporal Features**: Incorporate time-series data if available
5. **Explainability**: Add Grad-CAM visualization for image regions influencing predictions

