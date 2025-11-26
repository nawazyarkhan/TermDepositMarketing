# Term deposit Marketing
this project is about analyzing customer data on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.
This is about developing a robust machine learning system that leverages information coming from call center data.
Ultimately, Looking for ways to improve the success rate for calls made to customers for any product that our clients offer. Towards this goal we are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

## Data description:
The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

## Attributes

- age : age of customer (numeric)
- job : type of job (categorical)
- marital : marital status (categorical)
- education (categorical)
- default: has credit in default? (binary)
- balance: average yearly balance, in euros (numeric)
- housing: has a housing loan? (binary)
- loan: has personal loan? (binary)
- contact: contact communication type (categorical)
- day: last contact day of the month (numeric)
- month: last contact month of year (categorical)
- duration: last contact duration, in seconds (numeric)
- campaign: number of contacts performed during this campaign and for this client (numeric, -includes last contact)

- Output (desired target):
y - has the client subscribed to a term deposit? (binary)

> **Repository structure begins here:*  
> ```text
> ├── .gitignore                   # Git ignore file
> ├── data/                        # raw & processed data
> │   └── raw/
> │       └── term-deposit-marketing-2020.csv
> ├── notebooks/
> │   └── ExplorationNModeling/
> │       └── TDM.ipynb            # notebook covering exploration & modelling
> ├── reports/
> │   └── figures/                 # generated figures and plots
> ├── setup/
> │   └── requirements.txt         # Python dependencies
> └── README.md                    # this file
> ```  *


# Term Deposit Prediction & ML Platform  
*Automating Term-Deposit Subscription Prediction using LightGBM & Data Engineering Best Practices*

## 🚀 Project Summary  
In this project, I developed an end-to-end machine learning solution designed to predict whether customers will subscribe to a term deposit — a core business metric for our banking/client. The project harnessed the power of feature engineering, advanced encoding techniques, and model pipelines using LightGBM , Decision Tree, Logistic regression, XGboost, RandomForest where LightGBM outperfromed as compared to other models.

During my role at Apziva, I took ownership of the full workflow: from data ingestion and exploratory modelling (via the notebook `TDM.ipynb`) through to model evaluation and deployment readiness. This solution addresses imbalanced class distribution, accounts for categorical and cyclic variables (such as the “month” field), and generates actionable insights for the business.

## 🧭 Project Flow / Step-by-Step  

### 1. Data Acquisition & Understanding  
- Loaded the bank-marketing / term deposit dataset (source: internal CSV) containing customer attributes (age, job, marital status, education, contact, month, campaign, etc.).  
- Inspected the target variable: subscription to term deposit (`yes`/`no`).  
- Assessed class imbalance and feature distributions.

### 2. Exploratory Data Analysis & Pre-processing  
- Conducted summary statistics, missing value detection, and visualised key patterns (e.g., contact type, campaign effect, month distribution).  
- Cleaned and standardised numerical and categorical features.  
- Treated features like `month`  and others (e.g., `contact`, `job`, `housing` etc) via one-hot encoding to ensure correct modelling semantics.

### 3. Feature Engineering  
- Created new features capturing domain insights (for example, interaction between campaign count and previous contacts).  
- Encoded categorical variables in a scalable manner using `OneHotEncoder(handle_unknown='ignore')` to accommodate unseen categories in production.  
- Scaled numerical features via standardisation to enhance model convergence and stability.

### 4. Model Construction & Training  
- Set up the LightGBM classifier with an initial baseline, then tuned hyperparameters (learning rate, num_leaves, min_data_in_leaf, feature_fraction, bagging_fraction, etc.) to maximise ROC-AUC while controlling for overfitting.  
- Utilised a stratified train/validation/test split to fairly assess model performance given the class imbalance.  
- Applied early stopping and evaluation on the validation set to select the best iteration.

### 5. Evaluation & Results  
- Measured key performance metrics: ROC-AUC, accuracy, precision/recall for the minority class (subscribers).  
- Visualised feature importance and confusion matrix to interpret which variables were most predictive.  
- Demonstrated that the tuned model significantly outperformed baseline logistic regression, achieving strong predictive performance in the imbalanced classification context.

### 6. Deployment Readiness & Delivery  
- Packaged the preprocessing and modelling steps into reusable scripts/notebooks that can be run end-to-end (data load → preprocess → train → evaluate → predict).  
- Provided clear instructions and modular code structure to enable future retraining, monitoring, and inference workflows.

### 7. Clustering and customer segmentation 
 - The notebook explores unsupervised learning to segment customers:
- 📌 Methods used:
    - Agglomerative Clustering
    - K-Means Clustering
    - Silhouette Scores to determine the best value of k
    - The clustering step groups subscribers into meaningful segments based on:
    - Age
    - Balance
    - Duration of last call
    - Loan status
    - Campaign calls
    - Contact method
    - Seasonality

### 8. Interpretation of Clusters

- Each final cluster is analyzed to understand:

        - Demographics

        - Financial behaviour

        - Loan status

        - Marketing responsiveness

        - Contact channel preference

## 🧩 Key Contributions & Highlights  
- Engineered and managed the full ML pipeline for a **business-critical predictive task** in the banking domain.  
- Developed cyclic encoding for time-dependent features (e.g., month) to capture periodic behaviour.  
- Implemented robust categorical encoding with unseen-category handling (`handle_unknown='ignore'`), improving production readiness.  
- Optimised hyperparameters specifically for an imbalanced classification problem, improving model performance significantly. 
- catboost has given the best performance after hyperparameter tuning.  
- provide a summary on customer segmentation who subscribed the term deposit using k-mean clustering algorithm 
- Delivered documentation, modular code, and evaluation artifacts ready for hand-off to production or stakeholders.

## 🎯 Outcome: Customer Segmentation

- The workflow results in a set of distinct subscriber clusters, such as:

    - Senior High-Balance Savers
    - Young Mobile-First Investors
    - Mid-Age Mortgage Holders
    - Debt-Free Professionals
    - High-Risk Borrowers

## 🚀 Business Insights Delivered

- Identified high-value, low-risk depositor segments
- Detected which groups require more follow-up
- Mapped customer behaviour to marketing touchpoints
- Provided segmentation to support personalised outreach
- Highlighted groups likely to produce highest ROI

## 🎯 Concluding Remarks 
This project underscores my ability to deliver **end-to-end machine learning solutions** that translate into measurable business value. I took initiative from data exploration through modelling and evaluation and demonstrating proficiency in handling imbalanced data, engineering meaningful features, and tuning advanced tree-based models along with clustering / segmentation of subscribed customers

Please feel free to reachout for any queries! 

## 🛠 Reproducibility — Setup & Run

**Recommended:** use the provided conda environment.

```bash
# Create environment from shipped file
conda env create -f environment.yml
conda activate TDM

# OR using pip
conda create -n TDM python=3.10 -y
conda activate TDM
pip install -r setup/requirements.txt
```

**Run the analysis**
1. Place `term-deposit-marketing-2020.csv` inside the `data/raw` folder.
2. Open the notebook `notebooks/ExplorationNModeling/TDM.ipynb` and run cells sequentially.
3. Generated outputs (figures) will be saved to `reports/`.


---



## 📄 Acknowledgements  

Data source: Bank Marketing dataset (UCI / client internal data) — thanks to the team at Apziva for providing access.
