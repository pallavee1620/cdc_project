# Satellite Imagery-Based Property Valuation

A **Multimodal Regression Pipeline** that predicts property market value using both tabular data and satellite imagery. This project combines traditional real estate features with visual environmental context extracted from satellite images to improve property valuation accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Deliverables](#deliverables)

## 🎯 Overview

This project addresses the challenge of property valuation by integrating two different data modalities:
- **Tabular Data**: Traditional real estate features (bedrooms, bathrooms, square footage, location, etc.)
- **Satellite Imagery**: Visual environmental context captured from satellite images

The goal is to build a model that accurately values properties by incorporating "curb appeal" and neighborhood characteristics (like green cover, road density, proximity to water) that are visible in satellite imagery but not captured in traditional tabular features.

## 📁 Project Structure

```
p_cdc/
├── data/
│   ├── train(1)(train(1)).csv      # Training dataset
│   └── test2.xlsx                   # Test dataset
├── property_images_v2/              # Downloaded satellite images (training)
├── test_im/                         # Downloaded satellite images (test)
├── main.ipynb                       # Main training and inference notebook
├── preprocessing.ipynb             # Data preprocessing, EDA, and feature engineering
├── comparison.ipynb                 # Tabular-only vs Multimodal model comparison
├── data_fetcher.py                 # Script to download satellite images
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
```bash
conda create -n property_valuation python=3.9
conda activate property_valuation
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install the following packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install torch torchvision
pip install xgboost catboost
pip install pillow opencv-python
pip install tqdm requests
pip install jupyter notebook
```

## 🚀 Usage

### Step 1: Download Satellite Images

First, download satellite images for all properties using the data fetcher script:

```bash
python data_fetcher.py --csv_path "data/train(1)(train(1)).csv" --output_dir "property_images_v2"
```

**Parameters:**
- `--csv_path`: Path to your training CSV file
- `--output_dir`: Directory to save downloaded images
- `--zoom`: Zoom level (15-19, default: 19 for high detail)
- `--threads`: Number of parallel downloads (default: 20)

**Note:** The script uses the ESRI World Imagery tile service, which is free and doesn't require an API key.

### Step 2: Data Preprocessing and EDA

Run the preprocessing notebook to explore the data and prepare features:

```bash
jupyter notebook preprocessing.ipynb
```

This notebook includes:
- Data loading and initial exploration
- Exploratory Data Analysis (EDA)
- Data cleaning and outlier removal
- Feature engineering (geospatial, temporal, ratios)
- Feature standardization
- Data validation and export

### Step 3: Model Training

The main training pipeline is in `main.ipynb`:

```bash
jupyter notebook main.ipynb
```

This notebook:
- Loads and preprocesses the data
- Extracts visual features from satellite images using a CNN (ResNet18)
- Trains a hybrid ensemble model combining tabular and visual features
- Generates predictions on the test set

### Step 4: Model Comparison

Compare tabular-only vs multimodal approaches:

```bash
jupyter notebook comparison.ipynb
```

This notebook:
- Trains a model using only tabular features
- Trains a model using tabular + satellite image features
- Compares performance metrics (R², MAE, RMSE)
- Visualizes prediction accuracy and error distributions

## 📊 Methodology

### Data Pipeline

1. **Image Acquisition**: Satellite images are downloaded using latitude/longitude coordinates via the ESRI World Imagery tile service
2. **Visual Feature Extraction**: A pre-trained ResNet18 CNN extracts high-dimensional visual embeddings from satellite images
3. **Feature Engineering**: 
   - Log transformations for skewed features
   - Geospatial features (distance to center, distance to luxury hub)
   - Temporal features (house age, renovation status)
   - Ratio features (price per sqft, living ratio, etc.)
4. **Model Training**: Ensemble model combining Random Forest, XGBoost, and CatBoost

### Model Architecture

The final model uses a **Voting Regressor** ensemble:
- **Random Forest** (20% weight): Captures non-linear relationships
- **XGBoost** (40% weight): Gradient boosting for complex patterns
- **CatBoost** (40% weight): Handles categorical features effectively

**Features Used:**
- Tabular: bedrooms, bathrooms, sqft_living, floors, waterfront, view, grade, lat, long, etc.
- Visual: `visual_score` extracted from satellite images using CNN

### Key Features

- **Geospatial Analysis**: Distance from dataset center, distance to luxury property hub
- **Visual Context**: CNN-extracted features capturing environmental characteristics
- **Feature Engineering**: Log transformations, ratios, and interaction features
- **Ensemble Learning**: Combining multiple algorithms for robust predictions

## 📈 Results

### Model Performance

The multimodal approach (Tabular + Satellite Images) achieves:
- **R² Score**: ~0.895
- **MAE**: ~$67,000
- **RMSE**: Lower than tabular-only baseline

### Comparison: Tabular vs Multimodal

| Metric | Tabular Only | Multimodal | Improvement |
|--------|--------------|------------|-------------|
| R² Score | ~0.87 | ~0.90 | +3-4% |
| MAE | ~$73,000 | ~$67,000 | ~8% reduction |
| RMSE | Higher | Lower | Significant improvement |

**Key Insight**: Incorporating satellite imagery provides valuable environmental context that improves property valuation accuracy.

## 📦 Deliverables

### 1. Prediction File
- `submission_v2_unique.csv`: Final price predictions on test dataset
- Format: `id, predicted_price`

### 2. Code Repository
- `data_fetcher.py`: Script to download satellite images from API
- `preprocessing.ipynb`: Data cleaning, EDA, and feature engineering
- `main.ipynb`: Main training and inference pipeline
- `comparison.ipynb`: Tabular-only vs Multimodal model comparison
- `README.md`: Project documentation (this file)

### 3. Project Report
See the generated notebooks for:
- **Overview**: Approach and modeling strategy
- **EDA**: Visualizations of price distribution, correlations, and sample satellite images
- **Financial/Visual Insights**: Analysis of visual features driving value
- **Architecture**: How CNN features are integrated with tabular data
- **Results**: Performance comparison between tabular-only and multimodal approaches

## 🔍 Key Insights

1. **Visual Features Matter**: Satellite imagery captures environmental context (green cover, proximity to water, neighborhood density) that tabular features cannot represent.

2. **Geospatial Features**: Distance to luxury hubs and dataset center are strong predictors of property value.

3. **Feature Engineering**: Log transformations and ratio features significantly improve model performance.

4. **Ensemble Approach**: Combining multiple algorithms (RF, XGBoost, CatBoost) provides robust and accurate predictions.

## 🛠️ Technical Stack

- **Data Handling**: Pandas, NumPy, GeoPandas
- **Deep Learning**: PyTorch, torchvision (ResNet18)
- **Image Processing**: PIL, OpenCV
- **Machine Learning**: Scikit-learn, XGBoost, CatBoost
- **Visualization**: Matplotlib, Seaborn

## 📝 Notes

- The project uses the ESRI World Imagery tile service, which is free and doesn't require API keys
- Images are downloaded at zoom level 19 for high detail
- The model uses log-transformed prices for better convergence
- All preprocessing steps are designed to avoid data leakage

## 🤝 Contributing

This is a project submission. For questions or improvements, please refer to the project guidelines.

## 📄 License

This project is for educational purposes as part of a Data Science assignment.

---

**Author**: Property Valuation Team  
**Date**: 2024  
**Project**: Satellite Imagery-Based Property Valuation

