# Credit-Card-Fraud-Detection

## 🛡️ Project Overview

This is a comprehensive machine learning system designed to detect fraudulent credit card transactions with high accuracy. The project implements and compares multiple ML algorithms to identify the most effective approach for fraud detection on imbalanced datasets.

## 🎯 Objectives

- **Primary Goal**: Detect fraudulent credit card transactions with enhanced accuracy using ML models
- **Data Challenge**: Handle severely imbalanced datasets typical in fraud detection scenarios
- **Model Comparison**: Evaluate performance of Logistic Regression, SVM, and K-NN algorithms
- **Insights Generation**: Perform in-depth EDA using Seaborn to identify significant fraud patterns

## 📊 Dataset

- **Source**: Credit Card Fraud Detection Dataset (Kaggle)
- **Original Size**: ~284,807 transactions
- **Features**: 30 anonymized features (V1-V28) + Time, Amount, Class
- **Target Variable**: Class (0 = Legitimate, 1 = Fraudulent)
- **Imbalance Ratio**: ~99.8% legitimate vs ~0.2% fraudulent transactions

## 🔧 Technical Approach

### Data Preprocessing
- **Imbalance Handling**: Undersampling technique to create balanced dataset
- **Final Dataset**: 984 transactions (492 legitimate + 492 fraudulent)
- **Train-Test Split**: 80-20 with stratified sampling

### Machine Learning Models
1. **Logistic Regression**
   - Solver: liblinear
   - Max iterations: 1000
   - Best for: Linear relationships and interpretability

2. **Support Vector Machine (SVM)**
   - Kernel: RBF (Radial Basis Function)
   - Best for: Complex non-linear patterns

3. **K-Nearest Neighbors (K-NN)**
   - Neighbors: 5
   - Best for: Local pattern recognition

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate (minimizing false alarms)
- **Recall**: Fraud detection rate (minimizing missed frauds)
- **F1-Score**: Balanced precision-recall measure

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

### Required Libraries
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `scikit-learn` - Machine learning algorithms
- `seaborn` - Statistical data visualization
- `matplotlib` - Plotting and visualization

### Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraudguard-ml
   cd fraudguard-ml
   ```

2. **Download the dataset**
   - Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place it in the project root directory

3. **Run the analysis**
   Execute the code snippets in the following order:
   - Data Import and Basic Analysis
   - Exploratory Data Analysis
   - Data Preprocessing
   - Model Training (LR, SVM, K-NN)
   - Model Evaluation and Comparison

## 📈 Results & Performance

### Model Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 0.9340 | 0.9121 | 0.9565 | 0.9338 |
| SVM | 0.9543 | 0.9348 | 0.9783 | 0.9561 |
| K-NN | 0.9289 | 0.8936 | 0.9783 | 0.9342 |

### Key Insights
- **Best Overall Performance**: SVM with RBF kernel
- **Highest Recall**: K-NN (97.83% fraud detection rate)
- **Best Balance**: SVM offers optimal precision-recall tradeoff
- **Fraud Patterns**: Identified through comprehensive EDA visualizations

## 🔍 Key Features

- **Comprehensive EDA**: Detailed exploratory analysis with Seaborn visualizations
- **Multiple ML Algorithms**: Comparison of three different approaches
- **Imbalanced Data Handling**: Proper techniques for skewed datasets
- **Performance Visualization**: Clear charts and confusion matrices
- **Detailed Metrics**: Complete evaluation with precision, recall, F1-score
- **Reproducible Results**: Fixed random states for consistent outcomes

## 🎯 Business Impact

- **Financial Loss Prevention**: Early fraud detection saves money
- **Customer Trust**: Reduced false positives improve user experience
- **Operational Efficiency**: Automated detection reduces manual review
- **Scalable Solution**: Framework adaptable to different financial datasets

## 🔮 Future Enhancements

- **Deep Learning Models**: Implementation of neural networks
- **Real-time Processing**: Stream processing capabilities
- **Feature Engineering**: Advanced feature creation techniques
- **Ensemble Methods**: Combining multiple models for better performance
- **API Development**: REST API for real-time fraud scoring
