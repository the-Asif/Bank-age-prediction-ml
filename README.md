# 🏦 Machine Learning Techniques for Age Prediction & Customer Segmentation
> **A Study on the Bank Marketing Dataset**  
> *BRAC University — CSE422: Artificial Intelligence Project*

---

## 📌 Project Summary

The **Bank Marketing Dataset** is traditionally used for binary classification to predict whether a customer will subscribe to a term deposit (`y`). In this project, we reframe the dataset into a **Continuous Regression Task** aimed at predicting customer **Age** using demographic, financial, and campaign-related attributes.

To complement supervised regression, we also apply **Unsupervised Machine Learning (K-Means Clustering & Principal Component Analysis)** to segment the customer base into three distinct demographic age brackets (*Young*, *Middle-aged*, and *Old*).

---

## ✨ Key Features & Highlights

- 🔄 **Problem Reframing**: Reframed binary marketing classification into continuous age estimation ($R^2$ regression metrics) and demographic clustering.
- 🧹 **Robust Data Preprocessing**:
  - **Multicollinearity Removal**: Addressed high inter-correlation among economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`).
  - **Skewness Reduction**: Applied $\log(1+x)$ (`log1p`) transformation to compress heavily skewed features like `previous`.
  - **Outlier-Resilient Scaling**: Applied `RobustScaler` to normalize numeric features.
  - **Categorical Encoding**: One-Hot Encoded nominal attributes (`job`, `marital`, `education`).
- 🤖 **Supervised Regression Comparative Study**:
  - **Linear Regression**
  - **Decision Tree Regressor**
  - **Multi-Layer Perceptron (MLP) Regressor**
- 📊 **Unsupervised Demographic Segmentation**:
  - **K-Means Clustering ($K=3$)** mapping clients to *Young*, *Middle-aged*, and *Old*.
  - **Principal Component Analysis (PCA)** for 2D visual projection of cluster boundaries and centroids.

---

## 📁 Repository Structure

```text
├── CSE422_Group4_23101147_23101198 (1).pdf    # Comprehensive Project Report
├── CSE422_Group4_23101147_23101198 (2).ipynb  # Complete Jupyter Notebook Implementation
└── README.md                                  # Project Documentation
```

---

## 📊 Dataset Specifications

- **Total Instances**: 41,118 samples
- **Initial Features**: 20 candidate attributes (9 numerical, 11 categorical)
- **Target Variable**: `age` (continuous numerical target ranging from 17 to 98 years; 78 unique values)

### Summary of Feature Selection & Transformations

| Feature Name | Type | Action / Method | Reason |
| :--- | :--- | :--- | :--- |
| `age` | Numerical | **Target Variable** | Continuous target for regression |
| `y`, `duration`, `campaign`, `contact`, `month`, `day_of_week`, `poutcome` | Various | **Dropped** | Near-zero correlation with age / irrelevant to target |
| `euribor3m`, `nr.employed` | Numerical | **Dropped** | High multicollinearity with `emp.var.rate` ($r > 0.90$) |
| `pdays` | Numerical | **Dropped** | Dominated by sentinel value `999` (skewness $\approx -4.92$) |
| `job`, `marital`, `education` | Categorical | **One-Hot Encoded** | High predictive signal for age |
| `default`, `housing`, `loan` | Binary | **Retained** | Binary categorical indicators |
| `previous` | Numerical | **$\log(1+x)$ + RobustScaler** | Skewness compression & scale normalization |
| `emp.var.rate`, `cons.price.idx`, `cons.conf.idx` | Numerical | **RobustScaler** | Scale normalization & outlier resistance |

---

## 🔬 Experimental Setup & Results

### Data Split
- **Training Set**: 28,831 samples (70%)
- **Testing Set**: 12,357 samples (30%)

### 1. Supervised Learning Performance

Models were evaluated using **$R^2$ Score** and **Mean Squared Error (MSE)**:

| Model | Hyperparameters / Architecture | Training $R^2$ | Testing $R^2$ | MSE |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** | Default Parameters | 0.40 | 0.40 | 64.90 |
| **Decision Tree Regressor** | `max_depth = 10` | 0.47 | 0.42 | 62.63 |
| **Neural Network (MLP)** 🏆 | `hidden_layer_sizes = (64, 32)` | **0.49** | **0.44** | **60.31** |

> **Key Takeaway**: The **Multi-Layer Perceptron (MLP)** achieved the highest predictive accuracy ($R^2 = 0.44$, $\text{MSE} = 60.31$), demonstrating that non-linear architectures better capture the underlying patterns between customer demographics and age.

---

### 2. Unsupervised Learning & Clustering

Using **K-Means Clustering ($K=3$)** and **PCA 2D Projection**:

- **Young**: 10,586 clients (~25.7%)
- **Middle-aged**: 19,279 clients (~46.8%)
- **Old**: 11,323 clients (~27.5%)

```text
Cluster Distribution:
  Middle-aged  ████████████████████ 19,279
  Old          ███████████ 11,323
  Young        ██████████ 10,586
```

---

## 🚀 Installation & Running the Code

### Prerequisites
Ensure you have **Python 3.8+** installed along with the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Execution Steps
1. Clone or download this repository:
   ```bash
   git clone https://github.com/the-Asif/Bank-age-prediction-ml.git
   cd bank-age-prediction
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `CSE422_Group4_23101147_23101198 (2).ipynb` and execute all cells.

---

## 👥 Authors & Metadata

* **Institution**: BRAC University
* **Course**: CSE422: Artificial Intelligence (Summer 2025)
* **Section**: 31
* **Group**: Group 3 / Group 4

### Team Members
- **Rahabar Islam** (Student ID: `23101147`)
- **Md Asif Ahemed** (Student ID: `23101198`)

---

## 📜 License
This project is developed for academic purposes under BRAC University's CSE422 course curriculum.
