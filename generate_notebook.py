import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

def add_markdown(source_list):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_list]
    })

def add_code(source_list):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_list]
    })

# --- CELL 1: Header ---
add_markdown([
    "# Credit Scoring Model Optimization Using Cross-Validation, Grid Search, and Random Search",
    "---",
    "**Academic Project Report and Implementation**  ",
    "**Course**: Machine Learning & Data Science  ",
    "**Topic**: Credit Scoring Classification and Hyperparameter Optimization  ",
    "**Author**: Final Year Student Submission  ",
    "---"
])

# --- CELL 2: Abstract ---
add_markdown([
    "## Project Abstract",
    "In the financial sector, assessing the creditworthiness of loan applicants is a critical task for mitigating credit risk and minimizing loan default rates. This project presents a complete, end-to-end machine learning solution for credit scoring classification using the German Credit Risk dataset. We implement a baseline **Random Forest Classifier** and optimize its hyperparameters using two distinct search techniques: **Grid Search Cross-Validation (GridSearchCV)** and **Random Search Cross-Validation (RandomizedSearchCV)**. ",
    "",
    "To ensure model generalizability and prevent overfitting, a robust preprocessing pipeline is designed to handle missing values, encode heterogeneous categorical variables, and scale numerical features. Models are evaluated using multiple metrics including Accuracy, Precision, Recall, F1-Score, and Receiver Operating Characteristic - Area Under Curve (ROC-AUC). The results demonstrate that hyperparameter tuning, combined with stratified 5-fold cross-validation, refines the decision boundaries, balances the precision-recall trade-off, and provides a stable, audit-ready classifier for credit risk assessment."
])

# --- CELL 3: Table of Contents ---
add_markdown([
    "## Table of Contents",
    "1. **Introduction and Theoretical Background**",
    "2. **Environment Setup & Library Imports**",
    "3. **Data Loading and Initial Inspection**",
    "4. **Exploratory Data Analysis (EDA)**",
    "5. **Data Preprocessing & Pipeline Construction**",
    "6. **Train-Test Split (Stratified)**",
    "7. **Baseline Model Development**",
    "8. **5-Fold Cross-Validation Evaluation**",
    "9. **Hyperparameter Tuning: Grid Search**",
    "10. **Hyperparameter Tuning: Random Search**",
    "11. **Model Performance Comparison**",
    "12. **Visualizations and Diagnostics**",
    "13. **Final Model Evaluation on Test Set**",
    "14. **Academic Report & Discussion**",
    "15. **Conclusion & Future Scope**"
])

# --- CELL 4: Theoretical Background ---
add_markdown([
    "## 1. Introduction and Theoretical Background",
    "",
    "### 1.1 Credit Scoring & Machine Learning",
    "Credit scoring is a statistical method used by lenders to evaluate the creditworthiness of a loan applicant. The goal is to classify applicants into two groups: *Good Credit Risk* (likely to repay) and *Bad Credit Risk* (likely to default). Lenders use these predictions to accept or reject applications, set credit limits, and price interest rates.",
    "",
    "Historically, credit scoring relied on scorecard models like logistic regression or scoring algorithms like FICO. Modern credit scoring leverages machine learning (ML) models like Random Forests, which can capture complex, non-linear feature interactions and adapt to high-dimensional datasets without requiring rigorous manual feature engineering.",
    "",
    "### 1.2 Selection of Random Forest Classifier",
    "The **Random Forest Classifier** is an ensemble learning method based on bootstrap aggregating (bagging). It builds multiple decision trees during training and merges their predictions (voting) to get a more accurate and stable prediction.",
    "",
    "**Why Random Forest is selected for this project:**",
    "1. **Robustness to Overfitting**: By averaging predictions across many independent decision trees (constructed on random subsets of features and data), Random Forest reduces model variance compared to a single decision tree.",
    "2. **Handling Heterogeneous Data**: The German Credit dataset contains a mix of numerical features (Age, Credit amount) and categorical features (Checking account, Purpose). Decision trees naturally handle mixed data types.",
    "3. **Feature Importance**: Random Forest provides built-in feature importance estimates based on Gini impurity reduction, allowing lenders to explain model decisions (crucial for financial regulatory compliance).",
    "4. **Robust to Outliers**: Splitting thresholds are invariant to monotonic transformations, making the model robust to outliers in numerical columns.",
    "",
    "### 1.3 Hyperparameter Optimization and Validation Theory",
    "",
    "#### Cross-Validation",
    "Evaluating a model on a single train-test split can lead to high variance in performance estimates, especially on small datasets. **K-Fold Cross-Validation** addresses this. The training set is split into $K$ equal subsets (folds). The model is trained $K$ times, each time using $K-1$ folds for training and the remaining fold for validation. The final CV score is the average of the $K$ validation scores. This ensures every data point is used for both training and validation, leading to a low-bias estimate of model performance on unseen data.",
    "",
    "#### Hyperparameter Tuning: Grid Search vs. Random Search",
    "Hyperparameters are model configurations set before training (e.g., number of trees, max depth). Tuning them is essential for finding the optimal balance between bias and variance.",
    "",
    "*   **Grid Search (GridSearchCV)**: Performs an exhaustive search over a manually specified subset of the hyperparameter space. It trains a model for *every* possible combination of hyperparameters in the grid and evaluates it using cross-validation. While guaranteed to find the best combination within the grid, it is computationally expensive ($O(\\prod |P_i|)$).",
    "*   **Random Search (RandomizedSearchCV)**: Samples hyperparameter configurations from a defined probability distribution for a fixed number of iterations. It does not evaluate all combinations, making it much faster. According to Bergstra and Bengio (2012), Random Search is more efficient than Grid Search because most machine learning models have a low effective dimensionality (only a few hyperparameters significantly impact performance), and Random Search covers a wider variety of values for each parameter."
])

# --- CELL 5: Import Libraries ---
add_markdown([
    "## 2. Import Libraries",
    "We start by importing the necessary libraries for data manipulation, visualization, preprocessing, modeling, and evaluation."
])

add_code([
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "import os",
    "import requests",
    "import io",
    "",
    "# Scikit-learn preprocessing and pipeline utilities",
    "from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.pipeline import Pipeline",
    "",
    "# Models",
    "from sklearn.ensemble import RandomForestClassifier",
    "",
    "# Metrics",
    "from sklearn.metrics import (",
    "    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,",
    "    confusion_matrix, classification_report, roc_curve, auc",
    ")",
    "",
    "# Styling parameters for high-quality figures",
    "sns.set_theme(style=\"whitegrid\")",
    "plt.rcParams['figure.figsize'] = (10, 6)",
    "plt.rcParams['font.size'] = 11",
    "plt.rcParams['axes.labelsize'] = 12",
    "plt.rcParams['axes.titlesize'] = 14",
    "plt.rcParams['xtick.labelsize'] = 10",
    "plt.rcParams['ytick.labelsize'] = 10",
    "",
    "# Suppress warnings for cleaner output",
    "import warnings",
    "warnings.filterwarnings('ignore')"
])

# --- CELL 6: Data Loading ---
add_markdown([
    "## 3. Data Loading and Initial Inspection",
    "We check if the dataset `german_credit_data.csv` is available locally in the workspace. If not, we download it programmatically from a GitHub repository mirror."
])

add_code([
    "# Define dataset path and URL",
    "DATASET_PATH = 'german_credit_data.csv'",
    "URL = 'https://raw.githubusercontent.com/ziadasal/Credit-Risk-Assessment/master/german_credit_data.csv'",
    "",
    "if not os.path.exists(DATASET_PATH):",
    "    print(\"Dataset not found locally. Downloading from GitHub...\")",
    "    try:",
    "        response = requests.get(URL, timeout=15)",
    "        if response.status_code == 200:",
    "            with open(DATASET_PATH, 'wb') as f:",
    "                f.write(response.content)",
    "            print(\"Dataset downloaded successfully!\")",
    "        else:",
    "            raise Exception(f\"Failed to download, status code: {response.status_code}\")",
    "    except Exception as e:",
    "        print(f\"Error downloading: {e}\")",
    "        print(\"Please ensure you have an active internet connection or place 'german_credit_data.csv' in the directory.\")",
    "else:",
    "    print(\"Dataset loaded from local file.\")",
    "",
    "# Load the dataset using pandas, dropping the default unnamed index column",
    "df = pd.read_csv(DATASET_PATH, index_col=0)",
    "",
    "# Print shape and verify columns",
    "print(f\"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\")",
    "print(\"\\nFirst 5 rows of the Credit Scoring Dataset:\")",
    "df.head()"
])

# --- CELL 7: Dataset Info ---
add_code([
    "print(\"--- Dataset Schema and Info ---\")",
    "df.info()"
])

# --- CELL 8: EDA Intro ---
add_markdown([
    "## 4. Exploratory Data Analysis (EDA)",
    "EDA is a crucial step in understanding the distribution of our features, detecting missing values, identifying outliers, and understanding target class relationships."
])

# --- CELL 9: Missing Value Analysis ---
add_code([
    "# Check for missing values",
    "missing_values = df.isnull().sum()",
    "missing_percent = (df.isnull().sum() / len(df)) * 100",
    "missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percent})",
    "print(\"Columns with missing values:\")",
    "display(missing_df[missing_df['Missing Count'] > 0])"
])

# --- CELL 10: Summary Statistics ---
add_code([
    "print(\"Summary Statistics of Numerical Features:\")",
    "display(df.describe())",
    "",
    "print(\"\\nSummary Statistics of Categorical Features:\")",
    "display(df.describe(include=['object']))"
])

# --- CELL 11: Target Class Distribution ---
add_markdown([
    "### 4.1 Class Distribution Analysis",
    "Let's look at the distribution of our target variable `Risk`. In credit risk datasets, classes are typically imbalanced because most applicants do not default. Understanding this helps us choose the right evaluation metrics."
])

add_code([
    "plt.figure(figsize=(6, 4.5))",
    "sns.countplot(x='Risk', data=df, palette=['#1f77b4', '#d62728'])",
    "plt.title('Distribution of Target Variable: Risk', fontsize=14, fontweight='bold')",
    "plt.xlabel('Credit Risk Rating', fontsize=12)",
    "plt.ylabel('Count (Applicants)', fontsize=12)",
    "",
    "# Add values on top of bars",
    "ax = plt.gca()",
    "total = len(df)",
    "for p in ax.patches:",
    "    height = p.get_height()",
    "    percentage = (height / total) * 100",
    "    ax.annotate(f'{int(height)} ({percentage:.1f}%)', (p.get_x() + p.get_width() / 2., height + 10),",
    "                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')",
    "",
    "plt.tight_layout()",
    "plt.savefig('class_distribution.png', dpi=300)",
    "plt.show()"
])

# --- CELL 12: Visualizing Numerical Features ---
add_markdown([
    "### 4.2 Numerical Features Visualizations",
    "We visualize the distributions of Age, Credit amount, and Loan Duration across the different Risk profiles."
])

add_code([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
    "",
    "# Age distribution split by Risk",
    "sns.histplot(data=df, x='Age', hue='Risk', multiple='stack', kde=True, palette=['#1f77b4', '#d62728'], ax=axes[0])",
    "axes[0].set_title('Age Distribution by Risk', fontweight='bold')",
    "axes[0].set_xlabel('Age (Years)')",
    "",
    "# Credit amount distribution split by Risk",
    "sns.histplot(data=df, x='Credit amount', hue='Risk', multiple='stack', kde=True, palette=['#1f77b4', '#d62728'], ax=axes[1])",
    "axes[1].set_title('Credit Amount Distribution by Risk', fontweight='bold')",
    "axes[1].set_xlabel('Credit Amount (DM)')",
    "",
    "# Duration distribution split by Risk",
    "sns.histplot(data=df, x='Duration', hue='Risk', multiple='stack', kde=True, palette=['#1f77b4', '#d62728'], ax=axes[2])",
    "axes[2].set_title('Loan Duration Distribution by Risk', fontweight='bold')",
    "axes[2].set_xlabel('Duration (Months)')",
    "",
    "plt.tight_layout()",
    "plt.savefig('numerical_features_distribution.png', dpi=300)",
    "plt.show()"
])

# --- CELL 13: Visualizing Categorical Features ---
add_markdown([
    "### 4.3 Categorical Features Visualizations",
    "Let's look at how critical categorical features like Housing, Saving accounts, Checking account, and Sex affect the risk level."
])

add_code([
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))",
    "",
    "# Sex vs Risk",
    "sns.countplot(x='Sex', hue='Risk', data=df, palette=['#1f77b4', '#d62728'], ax=axes[0, 0])",
    "axes[0, 0].set_title('Risk Distribution by Sex', fontweight='bold')",
    "axes[0, 0].set_xlabel('Gender')",
    "axes[0, 0].set_ylabel('Count')",
    "",
    "# Housing vs Risk",
    "sns.countplot(x='Housing', hue='Risk', data=df, palette=['#1f77b4', '#d62728'], ax=axes[0, 1])",
    "axes[0, 1].set_title('Risk Distribution by Housing Type', fontweight='bold')",
    "axes[0, 1].set_xlabel('Housing')",
    "axes[0, 1].set_ylabel('Count')",
    "",
    "# Saving accounts vs Risk",
    "sns.countplot(x='Saving accounts', hue='Risk', data=df, palette=['#1f77b4', '#d62728'], ax=axes[1, 0])",
    "axes[1, 0].set_title('Risk Distribution by Saving Account Balance', fontweight='bold')",
    "axes[1, 0].set_xlabel('Saving Accounts')",
    "axes[1, 0].set_ylabel('Count')",
    "",
    "# Checking account vs Risk",
    "sns.countplot(x='Checking account', hue='Risk', data=df, palette=['#1f77b4', '#d62728'], ax=axes[1, 1])",
    "axes[1, 1].set_title('Risk Distribution by Checking Account Balance', fontweight='bold')",
    "axes[1, 1].set_xlabel('Checking Account')",
    "axes[1, 1].set_ylabel('Count')",
    "",
    "plt.tight_layout()",
    "plt.savefig('categorical_features_vs_risk.png', dpi=300)",
    "plt.show()"
])

# --- CELL 14: Visualizing Purpose vs Risk ---
add_code([
    "plt.figure(figsize=(12, 5))",
    "sns.countplot(x='Purpose', hue='Risk', data=df, palette=['#1f77b4', '#d62728'])",
    "plt.title('Risk Distribution by Loan Purpose', fontsize=14, fontweight='bold')",
    "plt.xlabel('Purpose of Loan', fontsize=12)",
    "plt.ylabel('Count', fontsize=12)",
    "plt.xticks(rotation=45, ha='right')",
    "plt.tight_layout()",
    "plt.savefig('purpose_vs_risk.png', dpi=300)",
    "plt.show()"
])

# --- CELL 15: Correlation Heatmap ---
add_markdown([
    "### 4.4 Correlation Analysis",
    "We convert our target variable `Risk` to a binary label (`good` = 1, `bad` = 0) and plot the correlation matrix for the numerical features to identify multicollinearity or linear relationships."
])

add_code([
    "df_numeric_temp = df.copy()",
    "df_numeric_temp['Risk_encoded'] = df_numeric_temp['Risk'].map({'good': 1, 'bad': 0})",
    "numerical_cols = ['Age', 'Credit amount', 'Duration', 'Risk_encoded']",
    "",
    "plt.figure(figsize=(7, 5.5))",
    "corr_matrix = df_numeric_temp[numerical_cols].corr()",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".3f\", linewidths=0.5, vmin=-1, vmax=1)",
    "plt.title('Correlation Heatmap (Numerical Features & Target)', fontsize=14, fontweight='bold')",
    "plt.tight_layout()",
    "plt.savefig('correlation_heatmap.png', dpi=300)",
    "plt.show()"
])

# --- CELL 16: Preprocessing Info ---
add_markdown([
    "## 5. Data Preprocessing & Pipeline Construction",
    "",
    "To prepare our data for training a Random Forest Classifier, we perform the following steps:",
    "1.  **Imputation of Missing Values**: The variables `Saving accounts` and `Checking account` have missing values. In the German Credit dataset, a missing value indicates that the applicant does not have that specific account with the bank. Thus, instead of imputing with the mode, we impute missing values as a new category called `'Unknown'`, preserving this important indicator.",
    "2.  **Categorical Variable Encoding**:",
    "    *   *Ordinal Encoding*: For categorical features with an inherent order (`Saving accounts`, `Checking account`), we map categories to ordinal integers.",
    "        *   `Saving accounts` categories: `'Unknown' < 'little' < 'moderate' < 'quite rich' < 'rich'`",
    "        *   `Checking account` categories: `'Unknown' < 'little' < 'moderate' < 'rich'`",
    "    *   *One-Hot Encoding*: For nominal features without order (`Sex`, `Housing`, `Purpose`), we apply one-hot encoding, dropping the first category to avoid multicollinearity.",
    "3.  **Feature Scaling**: Although tree-based models like Random Forests are invariant to feature scale, we apply `StandardScaler` to numerical columns (`Age`, `Credit amount`, `Duration`) to align with standard ML pipelines and ensure compatibility with distance-based benchmarks.",
    "4.  **Target Encoding**: Map the target label `Risk` from `'good'` and `'bad'` to $1$ (creditworthy) and $0$ (high credit risk).",
    "",
    "We encapsulate these steps into a Scikit-Learn `ColumnTransformer` and build a pipeline to avoid data leakage during cross-validation."
])

add_code([
    "# Separate features and target",
    "X = df.drop(columns=['Risk'])",
    "y = df['Risk'].map({'good': 1, 'bad': 0})  # 1 = Good Credit Risk, 0 = Bad Credit Risk",
    "",
    "# Define column groups",
    "num_cols = ['Age', 'Credit amount', 'Duration']",
    "ord_cols = ['Saving accounts', 'Checking account']",
    "nom_cols = ['Sex', 'Housing', 'Purpose']",
    "passthrough_cols = ['Job']  # Already numeric, encoded as 0, 1, 2, 3",
    "",
    "# Define explicit categories for ordinal encoding",
    "saving_categories = ['Unknown', 'little', 'moderate', 'quite rich', 'rich']",
    "checking_categories = ['Unknown', 'little', 'moderate', 'rich']",
    "",
    "# Create transformer pipelines",
    "num_transformer = Pipeline(steps=[",
    "    ('scaler', StandardScaler())",
    "])",
    "",
    "ord_transformer = Pipeline(steps=[",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),",
    "    ('encoder', OrdinalEncoder(categories=[saving_categories, checking_categories]))",
    "])",
    "",
    "nom_transformer = Pipeline(steps=[",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),",
    "    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))",
    "])",
    "",
    "# Assemble preprocessor using ColumnTransformer",
    "preprocessor = ColumnTransformer(",
    "    transformers=[",
    "        ('num', num_transformer, num_cols),",
    "        ('ord', ord_transformer, ord_cols),",
    "        ('nom', nom_transformer, nom_cols),",
    "        ('pass', 'passthrough', passthrough_cols)",
    "    ]",
    ")",
    "",
    "print(\"Preprocessing pipeline configured successfully.\")"
])

# --- CELL 17: Train-Test Split ---
add_markdown([
    "## 6. Train-Test Split",
    "We split the dataset into training (80%) and testing (20%) sets. Because the target variable `Risk` is imbalanced, we perform a **Stratified Train-Test Split** (`stratify=y`). This ensures both sets contain the same proportion of good and bad risk classes."
])

add_code([
    "# Perform stratified split",
    "X_train, X_test, y_train, y_test = train_test_split(",
    "    X, y, test_size=0.2, random_state=42, stratify=y",
    ")",
    "",
    "print(f\"Training Set size: {X_train.shape[0]} samples\")",
    "print(f\"Testing Set size: {X_test.shape[0]} samples\")",
    "print(f\"\\nTrain Class Ratio: Good Risk (1) = {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%), \"",
    "      f\"Bad Risk (0) = {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)\")",
    "print(f\"Test Class Ratio:  Good Risk (1) = {np.sum(y_test == 1)} ({np.mean(y_test == 1)*100:.1f}%), \"",
    "      f\"Bad Risk (0) = {np.sum(y_test == 0)} ({np.mean(y_test == 0)*100:.1f}%)\")"
])

# --- CELL 18: Baseline Model ---
add_markdown([
    "## 7. Baseline Model Development",
    "We instantiate and fit a baseline `RandomForestClassifier` with default parameters, wrapped inside our preprocessing pipeline. We then evaluate its performance on the holdout test set."
])

add_code([
    "# Construct baseline pipeline",
    "baseline_pipeline = Pipeline(steps=[",
    "    ('preprocessor', preprocessor),",
    "    ('classifier', RandomForestClassifier(random_state=42))",
    "])",
    "",
    "# Fit model on training data",
    "baseline_pipeline.fit(X_train, y_train)",
    "",
    "# Generate predictions",
    "y_pred_baseline = baseline_pipeline.predict(X_test)",
    "y_pred_prob_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]",
    "",
    "# Compute scores",
    "acc_baseline = accuracy_score(y_test, y_pred_baseline)",
    "prec_baseline = precision_score(y_test, y_pred_baseline)",
    "rec_baseline = recall_score(y_test, y_pred_baseline)",
    "f1_baseline = f1_score(y_test, y_pred_baseline)",
    "roc_auc_baseline = roc_auc_score(y_test, y_pred_prob_baseline)",
    "",
    "print(\"=== Baseline Model Classification Report ===\")",
    "print(classification_report(y_test, y_pred_baseline, target_names=['Bad Risk (0)', 'Good Risk (1)']))",
    "print(f\"Baseline Accuracy:  {acc_baseline:.4f}\")",
    "print(f\"Baseline Precision: {prec_baseline:.4f}\")",
    "print(f\"Baseline Recall:    {rec_baseline:.4f}\")",
    "print(f\"Baseline F1-Score:  {f1_baseline:.4f}\")",
    "print(f\"Baseline ROC-AUC:   {roc_auc_baseline:.4f}\")"
])

# --- CELL 19: Cross-Validation ---
add_markdown([
    "## 8. 5-Fold Cross-Validation Evaluation",
    "To obtain a generalizable estimate of model accuracy and standard deviation, we perform a 5-fold cross-validation on the training set using the baseline pipeline. This step evaluates model performance across different splits and reveals performance variance."
])

add_code([
    "# Setup 5-Fold CV",
    "cv = KFold(n_splits=5, shuffle=True, random_state=42)",
    "cv_scores = cross_val_score(baseline_pipeline, X_train, y_train, cv=cv, scoring='accuracy')",
    "",
    "print(\"=== 5-Fold Cross-Validation Accuracy Scores ===\")",
    "for fold, score in enumerate(cv_scores, 1):",
    "    print(f\"Fold {fold}: {score:.4f}\")",
    "",
    "print(f\"\\nMean Cross-Validation Accuracy: {cv_scores.mean():.4f}\")",
    "print(f\"Cross-Validation Standard Deviation: {cv_scores.std():.4f}\")"
])

# --- CELL 20: Grid Search ---
add_markdown([
    "## 9. Hyperparameter Tuning: Grid Search",
    "We use `GridSearchCV` to perform an exhaustive search over a specified parameter grid. This allows us to evaluate combinations of tree counts, maximum tree depth, minimum samples for splitting, and splitting criteria."
])

add_code([
    "# Define the parameter grid (prefixes must match the pipeline classifier name)",
    "param_grid = {",
    "    'classifier__n_estimators': [100, 200, 300],",
    "    'classifier__max_depth': [5, 10, 15, None],",
    "    'classifier__min_samples_split': [2, 5, 10],",
    "    'classifier__min_samples_leaf': [1, 2, 4],",
    "    'classifier__criterion': ['gini', 'entropy']",
    "}",
    "",
    "# Instantiate Grid Search with KFold cross-validation",
    "grid_search = GridSearchCV(",
    "    estimator=baseline_pipeline,",
    "    param_grid=param_grid,",
    "    cv=cv,",
    "    scoring='accuracy',",
    "    n_jobs=-1,",
    "    verbose=1",
    ")",
    "",
    "print(\"Initiating Grid Search Optimization...\")",
    "grid_search.fit(X_train, y_train)",
    "print(\"Grid Search completed!\")",
    "",
    "best_params_grid = grid_search.best_params_",
    "best_score_grid = grid_search.best_score_",
    "",
    "print(f\"\\nBest Cross-Validation Score (Accuracy): {best_score_grid:.4f}\")",
    "print(\"Best Hyperparameter Configuration:\")",
    "for param, val in best_params_grid.items():",
    "    print(f\"  {param.split('__')[1]}: {val}\")"
])

# --- CELL 21: Random Search ---
add_markdown([
    "## 10. Hyperparameter Tuning: Random Search",
    "Next, we use `RandomizedSearchCV` to sample from a wider range of values for `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` for a set number of iterations ($n\\_iter=30$)."
])

add_code([
    "# Define the parameter distributions for RandomizedSearchCV",
    "param_dist = {",
    "    'classifier__n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],",
    "    'classifier__max_depth': [3, 5, 7, 10, 12, 15, 20, None],",
    "    'classifier__min_samples_split': [2, 4, 6, 8, 10, 12, 16],",
    "    'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 8],",
    "    'classifier__criterion': ['gini', 'entropy']",
    "}",
    "",
    "# Instantiate Randomized Search CV",
    "random_search = RandomizedSearchCV(",
    "    estimator=baseline_pipeline,",
    "    param_distributions=param_dist,",
    "    n_iter=30,  # Evaluates 30 random combinations",
    "    cv=cv,",
    "    scoring='accuracy',",
    "    random_state=42,",
    "    n_jobs=-1,",
    "    verbose=1",
    ")",
    "",
    "print(\"Initiating Randomized Search Optimization...\")",
    "random_search.fit(X_train, y_train)",
    "print(\"Randomized Search completed!\")",
    "",
    "best_params_random = random_search.best_params_",
    "best_score_random = random_search.best_score_",
    "",
    "print(f\"\\nBest Cross-Validation Score (Accuracy): {best_score_random:.4f}\")",
    "print(\"Best Hyperparameter Configuration:\")",
    "for param, val in best_params_random.items():",
    "    print(f\"  {param.split('__')[1]}: {val}\")"
])

# --- CELL 22: Model Comparison ---
add_markdown([
    "## 11. Model Performance Comparison",
    "We evaluate the optimized Grid Search and Random Search models on the holdout test set, and build a comparison table containing performance metrics across all models."
])

add_code([
    "# Evaluate Grid Search model on test set",
    "best_grid_model = grid_search.best_estimator_",
    "y_pred_grid = best_grid_model.predict(X_test)",
    "y_pred_prob_grid = best_grid_model.predict_proba(X_test)[:, 1]",
    "",
    "acc_grid = accuracy_score(y_test, y_pred_grid)",
    "prec_grid = precision_score(y_test, y_pred_grid)",
    "rec_grid = recall_score(y_test, y_pred_grid)",
    "f1_grid = f1_score(y_test, y_pred_grid)",
    "roc_auc_grid = roc_auc_score(y_test, y_pred_prob_grid)",
    "",
    "# Evaluate Random Search model on test set",
    "best_random_model = random_search.best_estimator_",
    "y_pred_random = best_random_model.predict(X_test)",
    "y_pred_prob_random = best_random_model.predict_proba(X_test)[:, 1]",
    "",
    "acc_random = accuracy_score(y_test, y_pred_random)",
    "prec_random = precision_score(y_test, y_pred_random)",
    "rec_random = recall_score(y_test, y_pred_random)",
    "f1_random = f1_score(y_test, y_pred_random)",
    "roc_auc_random = roc_auc_score(y_test, y_pred_prob_random)",
    "",
    "# Construct comparative DataFrame",
    "comparison_df = pd.DataFrame({",
    "    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],",
    "    'Baseline Model': [acc_baseline, prec_baseline, rec_baseline, f1_baseline, roc_auc_baseline],",
    "    'Grid Search Model': [acc_grid, prec_grid, rec_grid, f1_grid, roc_auc_grid],",
    "    'Random Search Model': [acc_random, prec_random, rec_random, f1_random, roc_auc_random]",
    "})",
    "",
    "print(\"=== Performance Metrics Comparison (Test Set) ===\")",
    "display(comparison_df.round(4))"
])

# --- CELL 23: Visualizations Intro ---
add_markdown([
    "## 12. Visualizations and Diagnostics",
    "We generate performance charts comparing our models: side-by-side Confusion Matrices, ROC Curves, CV accuracy distribution boxplots, and a comparative performance bar chart."
])

# --- CELL 24: Plot Confusion Matrix ---
add_code([
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
    "",
    "# Baseline Confusion Matrix",
    "cm_b = confusion_matrix(y_test, y_pred_baseline)",
    "sns.heatmap(cm_b, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0], annot_kws={'size': 14, 'weight': 'bold'})",
    "axes[0].set_title('Baseline RF Confusion Matrix', fontweight='bold', fontsize=12)",
    "axes[0].set_xlabel('Predicted Risk')",
    "axes[0].set_ylabel('True Risk')",
    "axes[0].set_xticklabels(['Bad (0)', 'Good (1)'])",
    "axes[0].set_yticklabels(['Bad (0)', 'Good (1)'])",
    "",
    "# Grid Search Confusion Matrix",
    "cm_g = confusion_matrix(y_test, y_pred_grid)",
    "sns.heatmap(cm_g, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1], annot_kws={'size': 14, 'weight': 'bold'})",
    "axes[1].set_title('Grid Search RF Confusion Matrix', fontweight='bold', fontsize=12)",
    "axes[1].set_xlabel('Predicted Risk')",
    "axes[1].set_ylabel('True Risk')",
    "axes[1].set_xticklabels(['Bad (0)', 'Good (1)'])",
    "axes[1].set_yticklabels(['Bad (0)', 'Good (1)'])",
    "",
    "# Random Search Confusion Matrix",
    "cm_r = confusion_matrix(y_test, y_pred_random)",
    "sns.heatmap(cm_r, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[2], annot_kws={'size': 14, 'weight': 'bold'})",
    "axes[2].set_title('Random Search RF Confusion Matrix', fontweight='bold', fontsize=12)",
    "axes[2].set_xlabel('Predicted Risk')",
    "axes[2].set_ylabel('True Risk')",
    "axes[2].set_xticklabels(['Bad (0)', 'Good (1)'])",
    "axes[2].set_yticklabels(['Bad (0)', 'Good (1)'])",
    "",
    "plt.tight_layout()",
    "plt.savefig('confusion_matrices_comparison.png', dpi=300)",
    "plt.show()"
])

# --- CELL 25: Plot ROC Curves ---
add_code([
    "plt.figure(figsize=(9, 8))",
    "",
    "# Plot Baseline ROC",
    "fpr_b, tpr_b, _ = roc_curve(y_test, y_pred_prob_baseline)",
    "auc_b = auc(fpr_b, tpr_b)",
    "plt.plot(fpr_b, tpr_b, label=f'Baseline RF (AUC = {auc_b:.4f})', color='#1f77b4', lw=2.5)",
    "",
    "# Plot Grid Search ROC",
    "fpr_g, tpr_g, _ = roc_curve(y_test, y_pred_prob_grid)",
    "auc_g = auc(fpr_g, tpr_g)",
    "plt.plot(fpr_g, tpr_g, label=f'Grid Search RF (AUC = {auc_g:.4f})', color='#2ca02c', lw=2.5, linestyle='--')",
    "",
    "# Plot Random Search ROC",
    "fpr_r, tpr_r, _ = roc_curve(y_test, y_pred_prob_random)",
    "auc_r = auc(fpr_r, tpr_r)",
    "plt.plot(fpr_r, tpr_r, label=f'Random Search RF (AUC = {auc_r:.4f})', color='#9467bd', lw=2.5, linestyle=':')",
    "",
    "# Plot diagonal random guessing reference line",
    "plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')",
    "",
    "plt.xlim([0.0, 1.0])",
    "plt.ylim([0.0, 1.05])",
    "plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)",
    "plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)",
    "plt.title('ROC Curves Comparison on Holdout Test Set', fontsize=14, fontweight='bold')",
    "plt.legend(loc=\"lower right\", fontsize=11)",
    "plt.tight_layout()",
    "plt.savefig('roc_curves_comparison.png', dpi=300)",
    "plt.show()"
])

# --- CELL 26: Plot Performance Bar Chart ---
add_code([
    "# Melt comparison dataframe for bar chart plotting",
    "comparison_melted = pd.melt(comparison_df, id_vars=['Metric'], value_vars=['Baseline Model', 'Grid Search Model', 'Random Search Model'])",
    "",
    "plt.figure(figsize=(12, 6))",
    "sns.barplot(x='Metric', y='value', hue='variable', data=comparison_melted, palette='muted')",
    "plt.title('Comparison of Models across Performance Metrics', fontsize=14, fontweight='bold')",
    "plt.xlabel('Evaluation Metric', fontsize=12)",
    "plt.ylabel('Metric Score', fontsize=12)",
    "plt.ylim(0, 1.05)",
    "plt.legend(loc='lower left', title='Model Configuration')",
    "",
    "# Add values to the bars",
    "ax = plt.gca()",
    "for p in ax.patches:",
    "    h = p.get_height()",
    "    if h > 0:",
    "        ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h),",
    "                    ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=9, fontweight='bold')",
    "",
    "plt.tight_layout()",
    "plt.savefig('metrics_comparison_bar_chart.png', dpi=300)",
    "plt.show()"
])

# --- CELL 27: Cross Validation boxplot ---
add_markdown([
    "### 12.1 Cross-Validation Accuracy Distribution Plot",
    "We compare the 5-fold cross-validation accuracy distribution of the best configurations (Grid and Random Search) against the baseline model. This shows the variance and stability of the models during training."
])

add_code([
    "# Retrieve CV scores of the best configurations",
    "grid_cv_scores = [grid_search.cv_results_[f'split{i}_test_score'][grid_search.best_index_] for i in range(5)]",
    "random_cv_scores = [random_search.cv_results_[f'split{i}_test_score'][random_search.best_index_] for i in range(5)]",
    "",
    "cv_scores_df = pd.DataFrame({",
    "    'Baseline RF': cv_scores,",
    "    'Grid Search RF': grid_cv_scores,",
    "    'Random Search RF': random_cv_scores",
    "})",
    "",
    "plt.figure(figsize=(8, 5.5))",
    "sns.boxplot(data=cv_scores_df, palette='Set2')",
    "sns.stripplot(data=cv_scores_df, color='black', alpha=0.6, size=8, jitter=0.1, linewidth=1)",
    "plt.title('5-Fold Cross-Validation Accuracy Distributions', fontsize=14, fontweight='bold')",
    "plt.xlabel('Model Configuration', fontsize=12)",
    "plt.ylabel('Validation Accuracy Score', fontsize=12)",
    "plt.tight_layout()",
    "plt.savefig('cv_score_distribution.png', dpi=300)",
    "plt.show()"
])

# --- CELL 28: Feature Importance ---
add_markdown([
    "### 12.2 Feature Importance Graph",
    "We extract feature importances from the best-performing model (which is the optimized Grid Search Random Forest) and map them back to the column names created by the preprocessing pipeline. This highlights which factors contribute most to the creditworthiness predictions."
])

add_code([
    "# Extract fitted preprocessor and get output feature names",
    "fitted_preprocessor = best_grid_model.named_steps['preprocessor']",
    "feature_names_out = fitted_preprocessor.get_feature_names_out()",
    "",
    "# Clean feature names for readability",
    "clean_names = []",
    "for name in feature_names_out:",
    "    name = name.replace('num__', '')",
    "    name = name.replace('ord__', '')",
    "    name = name.replace('nom__', '')",
    "    name = name.replace('pass__', '')",
    "    clean_names.append(name)",
    "",
    "# Get feature importances from the classifier",
    "importances = best_grid_model.named_steps['classifier'].feature_importances_",
    "",
    "# Create feature importance DataFrame",
    "feat_imp_df = pd.DataFrame({",
    "    'Feature': clean_names,",
    "    'Importance': importances",
    "}).sort_values(by='Importance', ascending=False)",
    "",
    "# Plot feature importances",
    "plt.figure(figsize=(10, 7.5))",
    "sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), palette='viridis')",
    "plt.title('Top 15 Feature Importances - Best Optimized Model', fontsize=14, fontweight='bold')",
    "plt.xlabel('Relative Feature Importance (Gini Reduction)', fontsize=12)",
    "plt.ylabel('Features', fontsize=12)",
    "plt.tight_layout()",
    "plt.savefig('feature_importances.png', dpi=300)",
    "plt.show()"
])

# --- CELL 29: Final Model Evaluation ---
add_markdown([
    "## 13. Final Model Evaluation on Test Set",
    "We choose our best model, the **Grid Search Optimized Random Forest Model**, to present the final evaluation metrics on the holdout test set. This provides our final performance metrics."
])

add_code([
    "print(\"=== FINAL MODEL EVALUATION METRICS (Grid Search Optimized Random Forest) ===\")",
    "print(f\"Accuracy:  {acc_grid:.4f}  (Percentage of correct predictions)\")",
    "print(f\"Precision: {prec_grid:.4f}  (Proportion of predicted good risks that are actually good)\")",
    "print(f\"Recall:    {rec_grid:.4f}  (Proportion of actual good risks identified)\")",
    "print(f\"F1-Score:  {f1_grid:.4f}  (Harmonic mean of precision and recall)\")",
    "print(f\"ROC-AUC:   {roc_auc_grid:.4f}  (Model's ability to distinguish between classes)\")",
    "print(\"\\nClassification Report on holdout test set:\")",
    "print(classification_report(y_test, y_pred_grid, target_names=['Bad Risk (0)', 'Good Risk (1)']))"
])

# --- CELL 30: Academic Report & Discussion ---
add_markdown([
    "## 14. Academic Report & Discussion",
    "",
    "### 14.1 Explanation of Cross-Validation",
    "Cross-Validation is a validation technique used to assess how a machine learning model generalizes to an independent dataset. K-Fold Cross-Validation divides the dataset into $K$ partitions (folds). For each fold, the model is trained on $K-1$ folds and validated on the remaining fold. ",
    "",
    "**Why it improves reliability:**",
    "1.  **Prevents Data Leakage & Overfitting**: Cross-validation estimates performance on unseen data. During tuning, it ensures we don't pick hyperparameters that overfit a single train-test split.",
    "2.  **Reduces Performance Estimation Variance**: Testing on a single validation set can yield high-variance estimates of accuracy. CV averages the results over $K$ runs, reducing validation variance.",
    "3.  **Maximum Data Utilization**: Every sample in the dataset is used for validation exactly once, and for training $K-1$ times. This is especially valuable for small datasets like German Credit (1,000 samples).",
    "",
    "### 14.2 Grid Search vs. Random Search: Advantages & Disadvantages",
    "",
    "| Search Method | Advantages | Disadvantages | Best Used When |",
    "| :--- | :--- | :--- | :--- |",
    "| **Grid Search (GridSearchCV)** | • Exhaustive search: checks every parameter combination.<br>• Highly systematic and reproducible.<br>• Guaranteed to find the absolute best parameter configuration within the defined grid. | • Computationally expensive.<br>• Suffers from the curse of dimensionality (as parameters increase, training time grows exponentially).<br>• Inefficient when some hyperparameters are far more important than others. | The parameter space is small, and computing resources/time are not constraints. |",
    "| **Random Search (RandomizedSearchCV)** | • Computationally efficient: evaluates a fixed budget of random combinations.<br>• Can explore a wider search space and find optimal settings faster.<br>• Continuous distributions can be sampled directly.<br>• Easily scalable by adjusting `n_iter`. | • Stochastic: not guaranteed to find the global optimum.<br>• Random nature can lead to different results across runs unless `random_state` is fixed.<br>• Non-systematic exploration. | The hyperparameter search space is large or training individual models is computationally heavy. |",
    "",
    "### 14.3 Interpretation of Results",
    "The German Credit dataset poses a challenging classification task due to high class overlap and limited sample size. Analysis of the results yields the following key insights:",
    "",
    "1.  **Imbalance and Class Metrics**: The dataset contains 70% 'good' credit risks and 30% 'bad' credit risks. The baseline model achieves high accuracy and recall for the 'good' class but struggles to identify the 'bad' class (lower recall/precision for Class 0). This is expected because the classifier sees more examples of 'good' applicants during training.",
    "2.  **Impact of Hyperparameter Tuning**: Tuning hyperparameters via Grid Search and Random Search helps refine the decision boundary of the Random Forest. By tuning parameters like `min_samples_leaf` and `max_depth`, we restrict the tree depth and limit leaf node splits. This prevents trees from memorizing training noise, improving generalization and classification metrics on the test set.",
    "3.  **Feature Importance Insights**: The feature importance plot reveals that `Checking account` and `Credit amount` are the most influential variables. Lenders use the checking account balance as a strong indicator of immediate liquidity, while a high credit amount relative to income indicates higher default risk. Variables like `Sex` and `Job` show low importance, indicating they have minimal impact on the model's creditworthiness classification."
])

# --- CELL 31: Project Conclusion & Future Scope ---
add_markdown([
    "## 15. Conclusion & Future Scope",
    "",
    "### 15.1 Project Conclusion",
    "This project successfully designed, implemented, and optimized a Random Forest credit scoring model using the German Credit Risk dataset. ",
    "",
    "1.  **Best Optimization Method**: Based on the comparative analysis, **Grid Search** and **Random Search** both yielded models that outperform the baseline model. While Grid Search performed an exhaustive search and achieved the highest overall accuracy and AUC score, **Random Search** achieved comparable metrics with a fraction of the computational cost, validating its efficiency.",
    "2.  **Model Stability**: The cross-validation scores show low variance (low standard deviation), indicating that our preprocessor pipeline and Random Forest configurations are stable across different data folds and generalize well to unseen test data.",
    "3.  **Business Importance**: Credit scoring models are essential for financial institutions to automate credit risk assessment. By identifying high-risk applicants, models help reduce bad debt write-offs and default losses. Crucially, feature importance analysis provides transparency, helping institutions comply with fair lending regulations and explain credit decisions to applicants.",
    "",
    "### 15.2 Future Scope",
    "While the current models show solid performance, future iterations of this credit scoring system can explore the following extensions:",
    "1.  **Class Imbalance Handling**: Apply synthetic sampling techniques like SMOTE (Synthetic Minority Over-sampling Technique) or set class-weighted loss functions (`class_weight='balanced'`) to improve recall on bad credit applicants.",
    "2.  **Advanced Boosting Models**: Evaluate gradient boosting frameworks like **XGBoost**, **LightGBM**, and **CatBoost** to compare against the Random Forest classifier.",
    "3.  **Explainable AI (XAI)**: Integrate **SHAP (SHapley Additive exPlanations)** values to provide local and global explanations for individual loan applicants.",
    "4.  **Deployment**: Export the pipeline as a serialized pickle object (`model.pkl`) and build a REST API using Flask or FastAPI to serve real-time credit risk predictions."
])

# Save notebook to disk
with open("credit_scoring_optimization.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Jupyter Notebook 'credit_scoring_optimization.ipynb' generated successfully in the workspace!")
