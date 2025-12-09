# 🏦 Term Deposit Marketing - ML Classification & Customer Segmentation

An end-to-end machine learning solution for predicting term deposit subscriptions and discovering customer segments using advanced classification algorithms and K-Means clustering.

## 📊 Project Overview

This project analyzes customer data from a European banking institution's direct marketing campaigns to:
1. **Predict** whether customers will subscribe to term deposits using multiple ML algorithms
2. **Segment** subscribers into distinct customer archetypes using unsupervised clustering
3. **Deliver** actionable insights for targeted marketing strategies

The solution leverages comprehensive data engineering, feature engineering, and model optimization techniques to address class imbalance and maximize business value.

## 📁 Data Description

The dataset originates from direct marketing campaigns conducted via phone calls. Campaign success is measured by term deposit subscriptions - short-term deposits with maturities ranging from one month to several years.

**Key Challenge:** Class imbalance (majority of customers do not subscribe)  
**Privacy:** All personally identifiable information has been removed

### 📋 Dataset Attributes

**Demographic Features:**
- `age`: Customer age (numeric)
- `job`: Type of occupation (categorical: admin, technician, services, management, etc.)
- `marital`: Marital status (categorical: married, single, divorced)
- `education`: Education level (categorical: primary, secondary, tertiary)

**Financial Features:**
- `balance`: Average yearly balance in euros (numeric)
- `default`: Has credit in default? (binary: yes/no)
- `housing`: Has housing loan? (binary: yes/no)
- `loan`: Has personal loan? (binary: yes/no)

**Campaign Features:**
- `contact`: Contact communication type (categorical: cellular, telephone, unknown)
- `day`: Last contact day of the month (numeric: 1-31)
- `month`: Last contact month of year (categorical: jan, feb, mar, etc.)
- `duration`: Last contact duration in seconds (numeric)
- `campaign`: Number of contacts during this campaign (numeric)

**Target Variable:**
- `y`: Has the client subscribed to a term deposit? (binary: yes/no)

---

## 🗂️ Repository Structure

```text
TermDepositMarketing/
├── .gitignore                          # Git ignore configuration
├── README.md                           # Project documentation (this file)
├── environment.yml                     # Conda environment specification
├── data/
│   ├── raw/
│   │   └── term-deposit-marketing-2020.csv    # Original dataset
│   └── processed/
│       └── subscribed_customers.csv           # Clustered subscriber data
├── notebooks/
│   └── ExplorationNModeling/
│       ├── TDMv1_3.ipynb               # main 
|── Utils
|     ├── tdm_utils.py                 # functions used 
├── reports/
│   └── figures/                        # Generated visualizations and plots
│       ├── elbow_method_kmeans.png
│       ├── silhouette_scores_kmeans.png
│       ├── pca_clusters_kmeans.png
│       └── [model evaluation plots...]
└── setup/
    └── requirements.txt                # Python dependencies
```

---

## 🎯 Project Objectives

### Part I: Predictive Modeling (Classification)
Build robust ML models to predict term deposit subscriptions with:
- **High accuracy** on imbalanced data
- **Interpretable features** for business insights
- **Production-ready pipelines** for deployment

### Part II: Customer Segmentation (Clustering)
Discover hidden customer archetypes among subscribers:
- **Identify distinct segments** using K-Means clustering
- **Profile each cluster** with demographic and behavioral patterns
- **Enable targeted marketing** strategies per segment

---

## 🚀 Machine Learning Pipeline

### 1️⃣ Data Acquisition & Understanding
- Loaded bank marketing dataset (`term-deposit-marketing-2020.csv`)
- **41,188 records** with 17 attributes
- Target variable: Binary classification (subscribe: yes/no)
- Assessed severe **class imbalance** (~11% positive class)

### 2️⃣ Exploratory Data Analysis (EDA)
- Comprehensive statistical analysis using `tdm_utils.py` functions
- Visualized feature distributions with `visualize_all_features()`
- Analyzed target variable correlations with `plot_feature_distribution_by_target()`
- Identified missing values and data quality issues
- Detected categorical features with 'unknown' values using `find_unknown_values()`

### 3️⃣ Feature Engineering
**Temporal Features:**
- Converted `day` → `week` bins using `convert_day_to_week()`
- Transformed `month` → `quarter` for seasonal patterns using `convert_month_to_quarter()`
- Created `is_quarter_end` binary flag for fiscal period targeting

**Financial Features:**
- Engineered `is_high_balance` indicator (above median balance)
- Created `loan_count` aggregating housing + personal loans
- Applied `balance_log` transformation to reduce skewness

**Behavioral Features:**
- Calculated `campaign_intensity` = duration / (campaign + 1)
- Created `contacted_multiple_times` binary flag
- Engineered `is_last_call_long` based on median duration

**Demographic Features:**
- Binned `age` into life stage groups (18-29, 30-39, 40-49, etc.)
- Created `education_job` interaction features

### 4️⃣ Data Preprocessing
- **Numerical features:** Standardized using `StandardScaler`
- **Categorical features:** One-hot encoded with `OneHotEncoder(handle_unknown='ignore')`
- **Target encoding:** Binary mapping using `encode_target_variable()`
- **Class weights:** Computed balanced weights using `compute_class_weights_balanced()`

### 5️⃣ Model Training & Evaluation

**Models Implemented:**
1. **Logistic Regression** - Linear baseline model
2. **Decision Tree** - Interpretable tree-based classifier
3. **Random Forest** - Ensemble bagging method
4. **XGBoost** - Gradient boosting champion
5. **CatBoost** - Category-optimized boosting (BEST PERFORMER)
6. **LightGBM** - Efficient gradient boosting
7. **K-Nearest Neighbors** - Instance-based learning

**Training Strategy:**
- Stratified train/validation/test split (70/15/15)
- Class weight balancing to address imbalance
- Hyperparameter tuning with GridSearchCV
- Early stopping to prevent overfitting

**Evaluation Metrics:**
- ROC-AUC (primary metric for imbalanced data)
- Precision, Recall, F1-Score
- Confusion Matrix visualization using `plot_confusion_matrix()`
- ROC Curves using `plot_roc_auc()`

**Best Model Performance:**
- **CatBoost** achieved highest ROC-AUC after hyperparameter optimization
- Strong performance on minority class (subscribers)
- Feature importance analysis revealed key predictors:
  - `duration` (call length)
  - `balance` (account balance)
  - `age` (customer life stage)
  - `campaign_intensity` (engagement quality)
  - `month`/`quarter` (seasonal timing)

### 6️⃣ Clustering & Segmentation

**Methodology:**
- Filtered dataset to **subscribers only** (~4,500 customers)
- Applied same feature engineering pipeline
- Scaled features using `StandardScaler`
- One-hot encoded categorical variables

**Optimal Cluster Selection:**
- **Elbow Method:** Tested k=1 to k=30, identified "elbow" at k=2-5
- **Silhouette Analysis:** Optimal silhouette score at **k=3**
- Final decision: **3 clusters** provide best balance of cohesion and separation

**K-Means Clustering (k=3):**
- Applied K-Means algorithm with `random_state=42`
- Generated cluster labels for all subscribers
- Visualized clusters using PCA (2D & 3D) projections

**Cluster Profiling:**
Used `get_clusters()` function to analyze each segment across:
- Demographics (age, job, marital status, education)
- Financial status (balance, loans, housing)
- Campaign behavior (duration, contacts, intensity)
- Contact preferences (cellular vs telephone)
- Temporal patterns (month, quarter, seasonality)

---

## 🎭 The Three Customer Tribes

### Cluster 0: The Established Professionals 🏆
**"Affluent Achievers"**

**Profile:**
- **Age:** Middle-aged (35-50 years)
- **Occupation:** Management, technicians, senior roles
- **Financial Status:** Highest account balances, minimal debt
- **Loans:** Low loan counts, financially independent
- **Campaign Response:** Moderate engagement, quality over quantity

**Strategic Approach:**
- Emphasize **wealth preservation** and **premium returns**
- Offer **high-balance tier products** with exclusive rates
- Deploy **relationship managers** for personalized service
- Target with **VIP treatment** and investment consultations
- Best candidates for **cross-selling** premium financial products

---

### Cluster 1: The Cautious Accumulators 💼
**"Prudent Savers"**

**Profile:**
- **Age:** Young to middle-aged (25-45 years)
- **Occupation:** Mixed professional backgrounds
- **Financial Status:** Moderate balances, building wealth
- **Loans:** Some loan activity, managing obligations
- **Campaign Response:** Balanced engagement, value-conscious

**Strategic Approach:**
- Focus on **safety, reliability, and steady growth**
- Offer **flexible term options** (6, 12, 24 months)
- Provide **financial education** content and tools
- Emphasize **lower minimum deposits** for accessibility
- Position as **stepping stones** to financial security
- Nurture for **future growth** as careers advance

---

### Cluster 2: The Emerging Subscribers 🌱
**"New Entrants"**

**Profile:**
- **Age:** Younger demographic (under 35)
- **Occupation:** Entry-level positions, students, early career
- **Financial Status:** Lower balances, building credit history
- **Loans:** Establishing financial footprint
- **Campaign Response:** **High engagement**, responsive to outreach

**Strategic Approach:**
- **Digital-first** engagement (mobile apps, social media)
- Offer **starter products** with low entry barriers
- Emphasize **financial literacy** and education
- Use **gamification** and milestone rewards
- Build **long-term loyalty** from early relationship
- **Lifetime value thinking:** Today's Cluster 2 becomes tomorrow's Cluster 0

---

## 📈 Business Impact & Insights

### Key Findings

1. **Segmentation Reveals Diversity:** Three distinct subscriber personas require differentiated strategies
2. **Balance is King:** Account balance is the strongest differentiator across all segments
3. **Age = Life Stage:** Younger subscribers behave fundamentally differently from established professionals
4. **Campaign Quality > Quantity:** `campaign_intensity` outperforms raw `campaign` count
5. **Timing Matters:** Seasonal patterns (quarters) influence subscription likelihood
6. **High Engagement ≠ High Value:** Cluster 2 has highest campaign response but lowest balances

### Actionable Recommendations

**Immediate Wins (0-3 months):**
- Launch **segment-specific email campaigns** using cluster profiles
- Create **three product tiers**: "Starter" (Cluster 2), "Builder" (Cluster 1), "Premium" (Cluster 0)
- Retrain marketing teams on the **three customer archetypes**
- Implement **targeted messaging** per segment

**Medium-term Strategy (3-12 months):**
- Develop **cluster transition models** (predict movement from Cluster 2 → Cluster 1 → Cluster 0)
- Implement **dynamic pricing** based on segment and predicted lifetime value
- Create **automated customer journeys** that evolve with cluster migration
- A/B test segment-specific campaigns to validate strategies

**Long-term Vision (12+ months):**
- Build **customer lifecycle platform** tracking segment movement
- Develop **predictive retention models** for each cluster
- Integrate clustering insights into **CRM systems** for real-time personalization
- Deploy **ML-powered recommendation engine** for next-best-action

---

## 🛠️ Technical Implementation

### Core Technologies
- **Python 3.10+** - Primary programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - ML algorithms and preprocessing
- **XGBoost, LightGBM, CatBoost** - Advanced gradient boosting
- **Matplotlib, Seaborn, Plotly** - Data visualization
- **Jupyter Notebook** - Interactive analysis environment

### Custom Utility Module: `tdm_utils.py`

**Data Exploration Functions:**
- `identify_feature_types()` - Separate categorical and numerical features
- `find_unknown_values()` - Detect missing/unknown categorical values
- `calculate_class_distribution()` - Analyze target variable balance
- `visualize_all_features()` - Create comprehensive EDA visualizations
- `plot_individual_features()` - Generate feature-specific plots
- `plot_feature_distribution_by_target()` - Compare distributions by target class

**Feature Engineering Functions:**
- `encode_target_variable()` - Binary encoding with custom mapping
- `convert_day_to_week()` - Temporal feature transformation
- `convert_month_to_quarter()` - Seasonal aggregation
- `compute_class_weights_balanced()` - Calculate balanced class weights

**Model Training & Evaluation:**
- `fit_model()` - Train model with preprocessor pipeline
- `evaluate_model()` - Comprehensive model evaluation with metrics
- `plot_confusion_matrix()` - Visualize classification performance
- `plot_roc_auc()` - Generate ROC curves and calculate AUC
- `calculate_auc_score()` - Compute ROC-AUC metric

**Clustering Functions:**
- `get_clusters()` - Generate and analyze K-Means clusters with profiling

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or pip

### Installation

**Option 1: Using Conda (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd TermDepositMarketing

# Create environment from specification
conda env create -f environment.yml
conda activate TDM
```

**Option 2: Using pip**
```bash
# Create virtual environment
conda create -n TDM python=3.10 -y
conda activate TDM

# Install dependencies
pip install -r setup/requirements.txt
```

### Running the Analysis

1. **Prepare Data:**
   ```bash
   # Ensure dataset is in the correct location
   data/raw/term-deposit-marketing-2020.csv
   ```

2. **Execute Main Notebook:**
   ```bash
   # Launch Jupyter
   jupyter notebook
   
   # Open and run:
   notebooks/ExplorationNModeling/TDMv1_3.ipynb
   ```

3. **Run All Cells Sequentially:**
   - The notebook is designed to be executed from top to bottom
   - Intermediate results are saved automatically
   - Figures are exported to `reports/figures/`
   - Clustered data saved to `data/processed/subscribed_customers.csv`

### Using Utility Functions

```python
# Import utilities
import sys
sys.path.append('notebooks/ExplorationNModeling/')
from tdm_utils import *

# Load data
import pandas as pd
df = pd.read_csv('data/raw/term-deposit-marketing-2020.csv', sep=';')

# Identify feature types
categorical_features, numerical_features = identify_feature_types(df)

# Encode target variable
df['y_encoded'] = encode_target_variable(df, target_column='y')

# Compute class weights
class_weights = compute_class_weights_balanced(df['y_encoded'])

# Visualize features
visualize_all_features(df, categorical_features, numerical_features)
```

---

## 🧩 Key Highlights & Contributions

- ✅ Engineered **end-to-end ML pipeline** for business-critical banking prediction task
- ✅ Developed comprehensive **feature engineering** with temporal, financial, and behavioral features
- ✅ Implemented robust **categorical encoding** with `handle_unknown='ignore'` for production readiness
- ✅ Optimized **7 different ML models** with hyperparameter tuning for imbalanced classification
- ✅ **CatBoost achieved best performance** after extensive hyperparameter optimization
- ✅ Discovered **3 distinct customer segments** using K-Means clustering with optimal k selection
- ✅ Created **reusable utility module** (`tdm_utils.py`) with 19 custom functions
- ✅ Delivered **actionable business insights** with segment-specific marketing strategies
- ✅ Built **production-ready pipelines** with modular code and comprehensive documentation

---

## 📊 Results Summary

### Classification Performance
- **Best Model:** CatBoost
- **Metric:** ROC-AUC optimized for imbalanced data
- **Key Features:** duration, balance, age, campaign_intensity, temporal patterns
- **Production Ready:** Pipeline with preprocessing, encoding, and model inference

### Clustering Results
- **Optimal Clusters:** k=3 (validated by Elbow + Silhouette methods)
- **Segments:** Established Professionals, Cautious Accumulators, Emerging Subscribers
- **Business Value:** Targeted marketing strategies per archetype
- **Visualization:** PCA projections (2D & 3D) showing clear separation

---

## 🎯 Concluding Remarks

This project demonstrates end-to-end machine learning expertise in:
- **Data Engineering:** Feature engineering, encoding, scaling, imbalance handling
- **Model Development:** Multiple algorithms, hyperparameter tuning, ensemble methods
- **Unsupervised Learning:** Clustering, optimal k selection, segment profiling
- **Business Translation:** Converting technical insights into actionable strategies
- **Code Quality:** Modular utilities, comprehensive documentation, reproducible pipelines

The solution is production-ready and delivers measurable business value through predictive models and customer segmentation.

---

## 📄 Acknowledgements


**Project Context:** Developed as part of data science portfolio at Apziva

---

## 📧 Contact

For questions, collaborations, or inquiries about this project:
- **GitHub:** [https://github.com/nawazyarkhan]
- **LinkedIn:** [https://www.linkedin.com/in/nawaz-yar-khan/]
- **Email:** [nawazyarkhan@gmail.com]

---

**Built using Python, Scikit-learn, and advanced ML techniques**
