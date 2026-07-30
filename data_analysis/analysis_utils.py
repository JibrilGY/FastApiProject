import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures


# 1. General Structure, Types and Missing Value Analysis
def check_general_structure(df):
  print("=== Data Types and Non-Null Information ===")
  print(df.info())
  print("\n=== Missing Value Summary ===")
  missing = df.isnull().sum()
  missing_pct = (missing / len(df)) * 100
  missing_df = pd.DataFrame(
      {"Missing Count": missing, "Missing Percentage (%)": missing_pct}
  )
  print(missing_df[missing_df["Missing Count"] > 0])
  return missing_df


# 2. Statistical Distribution
def plot_statistical_distributions(df, numerical_cols):
  for col in numerical_cols:
    if col in df.columns:
      fig, axes = plt.subplots(1, 2, figsize=(12, 4))
      sns.histplot(df[col], kde=True, ax=axes[0], color="blue")
      axes[0].set_title(f"Distribution (Histogram): {col}")

      sns.boxplot(x=df[col], ax=axes[1], color="orange")
      axes[1].set_title(f"Outlier (Boxplot): {col}")
      plt.tight_layout()
      plt.show()


# 3. Target Variable Distribution
def plot_target_distribution(df, target_col):
  plt.figure(figsize=(6, 4))
  sns.countplot(data=df, x=target_col, palette="Set2", hue=target_col, legend=False)
  plt.title(f"Target Distribution: {target_col}")
  plt.show()


# 4. Coefficients, Correlation Curves + Target Relationship
def analyze_feature_correlations(df, target_col, corr_threshold=0.85):
  corr_matrix = df.select_dtypes(include=[np.number]).corr()

  if target_col in corr_matrix.columns:
    target_corr = (
        corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    )
    print("=== Correlation Ranking with Target ===")
    print(target_corr)
    print("-" * 40)
  else:
    target_corr = None

  plt.figure(figsize=(10, 8))
  sns.heatmap(
      corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
  )
  plt.title("Feature-to-Feature Correlation Matrix")
  plt.tight_layout()
  plt.show()

  upper_tri = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
  )
  to_drop = [
      col for col in upper_tri.columns if any(abs(upper_tri[col]) > corr_threshold)
  ]
  print(f"⚠️ Highly correlated (overlapping) features: {to_drop}")

  return target_corr, to_drop


# For Linear Models
def plot_model_coefficients(pipeline, feature_names, classifier_step="classifier"):
  classifier = pipeline.named_steps.get(classifier_step)
  if classifier is not None and hasattr(classifier, "coef_"):
    coefs = (
        classifier.coef_[0]
        if len(classifier.coef_.shape) > 1
        else classifier.coef_
    )
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
    coef_df["Abs_Coef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values(by="Abs_Coef", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Coefficient",
        y="Feature",
        data=coef_df,
        palette="coolwarm",
        hue="Coefficient",
        legend=False,
    )
    plt.title("Model Coefficients")
    plt.axvline(x=0, color="grey", linestyle="--")
    plt.tight_layout()
    plt.show()
    return coef_df
  else:
    print("❌ This model does not have a 'coef_' attribute.")
    return None


# For Tree Models
def plot_tree_importances(
    pipeline, feature_names, classifier_step="classifier"
):
  classifier = pipeline.named_steps.get(classifier_step)

  if classifier is not None and hasattr(classifier, "feature_importances_"):
    importances = classifier.feature_importances_
    imp_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    )
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=imp_df,
        palette="viridis",
        hue="Importance",
        legend=False,
    )
    plt.title("Random Forest - Feature Importances")
    plt.tight_layout()
    plt.show()
    return imp_df
  else:
    print("❌ This model does not have a 'feature_importances_' attribute.")
    return None


# 5. Feature Extraction - PCA
def extract_features_pca(X, n_components=2):
  pca = PCA(n_components=n_components)
  X_pca = pca.fit_transform(X.select_dtypes(include=[np.number]))

  pca_df = pd.DataFrame(
      X_pca, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index
  )

  print(f"=== PCA Feature Extraction ({n_components} Components) ===")
  print(
      f"Explained Variance Ratio Sum: {pca.explained_variance_ratio_.sum():.4f}"
  )
  return pca_df, pca


# 6. Feature Extraction - Polynomial and Interactions
def extract_polynomial_features(X, degree=2, interaction_only=False):
  num_X = X.select_dtypes(include=[np.number])
  poly = PolynomialFeatures(
      degree=degree, interaction_only=interaction_only, include_bias=False
  )
  X_poly = poly.fit_transform(num_X)
  feature_names = poly.get_feature_names_out(num_X.columns)

  poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

  print(
      f"=== Polynomial / Interaction Extraction (Degree: {degree}) ==="
      f" New Column Count: {poly_df.shape[1]}"
  )
  return poly_df, poly


# 7. Feature Selection
def select_best_features_anova(X, y, k=5):
  selector = SelectKBest(score_func=f_classif, k=k)
  selector.fit(X, y)
  scores_df = pd.DataFrame(
      {"Feature": X.columns, "Score": selector.scores_}
  ).sort_values(by="Score", ascending=False)

  print(f"=== Top {k} Features (ANOVA / SelectKBest) ===")
  print(scores_df.head(k))
  return scores_df["Feature"].head(k).tolist()


# 8. Correlation Filter
def correlation_filter(X_train, X_test, corr_threshold):
  corr_matrix = (
      X_train.select_dtypes(include=[np.number]).corr().abs()
  )

  upper_tri = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
  )
  to_drop_corr = [
      col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)
  ]

  X_train_reduced = X_train.drop(columns=to_drop_corr)
  X_test_reduced = X_test.drop(columns=to_drop_corr, errors="ignore")
  print(
      f"9. Remaining Feature Count After Correlation Filter:"
      f" {X_train_reduced.shape[1]}"
  )
  return X_train_reduced, X_test_reduced


# 9. Outlier Clipping
def apply_iqr_clipping(X_train, X_test, numerical_cols, factor=1.5):
  # 1. Create copies to preserve original data
  X_train_clipped = X_train.copy()
  X_test_clipped = X_test.copy()

  outlier_summary = {}

  for col in numerical_cols:
    if col in X_train.columns:
      Q1 = X_train[col].quantile(0.25)
      Q3 = X_train[col].quantile(0.75)
      IQR = Q3 - Q1

      # Calculate bounds strictly based on X_train
      lower_bound = Q1 - (factor * IQR)
      upper_bound = Q3 + (factor * IQR)

      # Keep outlier counts for reporting
      outliers = X_train[
          (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
      ]
      outlier_summary[col] = len(outliers)

      # Apply clipping inside the loop for each column
      X_train_clipped[col] = X_train_clipped[col].clip(
          lower=lower_bound, upper=upper_bound
      )
      X_test_clipped[col] = X_test_clipped[col].clip(
          lower=lower_bound, upper=upper_bound
      )

  # Reporting
  print("=== Outlier Counts ===")
  for col, count in outlier_summary.items():
    if count > 0:
      print(f"🔸 {col}: Found {count} outliers.")

  print(f"8. IQR Clipping Process Completed (Factor: {factor}).")
  return X_train_clipped, X_test_clipped, outlier_summary


def load_and_preprocess_data(
    file_path, target_col, test_size=0.2, random_state=42
):
  df = pd.read_csv(file_path)
  X = df.drop(target_col, axis=1)
  y = df[target_col]

  X = pd.get_dummies(X, drop_first=True)
  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y)

  X_train, X_test, y_train, y_test = train_test_split(
      X, y_encoded, test_size=test_size, random_state=random_state
  )
  print(f"'{file_path}' successfully loaded and processed!")
  print(f"Train Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

  return X_train, X_test, y_train, y_test, label_encoder


def handle_categorical_features(df, target_col=None):
  # If target column is specified, separate it to prevent transformation interference
  if target_col and target_col in df.columns:
    y = df[target_col]
    X = df.drop(columns=[target_col])
  else:
    X = df.copy()
    y = None

  # Identify categorical (string / object / category) columns
  categorical_cols = X.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()

  # Get numerical columns as well for informational purposes
  numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

  print(f"Detected numerical columns: {numeric_cols}")
  print(f"Detected categorical (string) columns: {categorical_cols}")

  # Apply One-Hot Encoding if categorical columns exist
  if categorical_cols:
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
  else:
    X_encoded = X.copy()

  # Return with target variable if it exists, otherwise return only X_encoded
  if y is not None:
    return X_encoded, y

  return X_encoded


def plot_categorical_distributions(df, categorical_cols):
  """Visualizes the frequency distribution (counts) of string/categorical

  columns in the dataset as a bar plot.
  """
  if not categorical_cols:
    print("No categorical columns found to visualize.")
    return

  num_cols = len(categorical_cols)
  # Create a dynamic subplot layout based on the number of columns
  ncols_grid = 2 if num_cols > 1 else 1
  nrows_grid = (num_cols + ncols_grid - 1) // ncols_grid

  fig, axes = plt.subplots(
      nrows=nrows_grid, ncols=ncols_grid, figsize=(12, 4 * nrows_grid)
  )

  # In case of a single grid, axes might not be an array, convert it to a list
  if num_cols == 1:
    axes = [axes]
  else:
    axes = axes.flatten()

  for i, col in enumerate(categorical_cols):
    sns.countplot(
        data=df,
        x=col,
        ax=axes[i],
        palette="Set2",
        order=df[col].value_counts().index,
    )
    axes[i].set_title(f"{col} - Distribution", fontsize=12, fontweight="bold")
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Person / Record Count", fontsize=10)
    axes[i].tick_params(axis="x", rotation=30)

  # Hide remaining empty subplots
  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()


def calculate_and_plot_mutual_information(df, target_col):
  """Calculates and visualizes the relationships between features and a

  categorical target variable using the Mutual Information method without
  training a model.
  """
  # Prepare categorical features for analysis using One-Hot Encoding
  X_processed, y_raw = handle_categorical_features(df, target_col=target_col)

  # Convert target variable to numeric code if it's a string
  le = LabelEncoder()
  y_encoded = le.fit_transform(y_raw)

  # Calculate Mutual Information scores
  mi_scores = mutual_info_classif(X_processed, y_encoded, random_state=42)
  mi_df = pd.DataFrame(
      {"Feature": X_processed.columns, "MI_Score": mi_scores}
  ).sort_values(by="MI_Score", ascending=False)

  print("--- Statistical Feature Importance Analysis (Mutual Information) ---")
  print(mi_df)

  # Visualization
  plt.figure(figsize=(10, 6))
  plt.barh(mi_df["Feature"], mi_df["MI_Score"], color="steelblue")
  plt.xlabel("Mutual Information Score", fontsize=11)
  plt.ylabel("Features", fontsize=11)
  plt.title(
      "Statistical Relationship Between Target and Features",
      fontsize=13,
      fontweight="bold",
  )
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.show()

  return mi_df


def analyze_mixed_feature_relations(df, target_col, threshold=0.01):
  """Calculates relationships with the target variable using Mutual

  Information in datasets containing both numerical and categorical (string)
  columns. Provides an alternative analysis without breaking the original
  correlation function.
  """
  # 1. Quantify categorical features using One-Hot Encoding
  X_processed, y_raw = handle_categorical_features(df, target_col=target_col)

  # 2. Convert target variable to numeric code if it's a string
  le = LabelEncoder()
  y_encoded = le.fit_transform(y_raw)

  # 3. Calculate statistical relationship scores
  mi_scores = mutual_info_classif(X_processed, y_encoded, random_state=42)

  target_corr = pd.Series(mi_scores, index=X_processed.columns).sort_values(
      ascending=False
  )

  # Identify weak columns falling below the threshold
  to_drop = target_corr[target_corr < threshold].index.tolist()

  print("--- Mixed/Categorical Feature Relationship Scores (Mutual Information) ---")
  print(target_corr)
  if to_drop:
    print(f"Low-score columns that can be dropped: {to_drop}")
  else:
    print("No weak columns found to drop, all features are strong.")

  return target_corr, to_drop


def select_best_features_anova_string(X, y, k=5):
  """Automatically quantifies the X matrix containing categorical/string columns

  (via One-Hot Encoding) and applies the ANOVA test.
  """
  # 1. Convert all internal string/categorical columns to numeric format (dummies)
  X_numeric = pd.get_dummies(X, drop_first=True)

  # Limit k if requested feature count exceeds total column count
  k = min(k, X_numeric.shape[1])

  # 2. Run SelectKBest and ANOVA test
  selector = SelectKBest(score_func=f_classif, k=k)
  selector.fit(X_numeric, y)

  # 3. Dump scores into a DataFrame
  scores_df = pd.DataFrame(
      {"Feature": X_numeric.columns, "Score": selector.scores_}
  ).sort_values(by="Score", ascending=False)

  print(f"=== Top {k} Features (ANOVA) ===")
  print(scores_df.head(k))

  # 4. Clean None/NaN values and convert to a list
  selected_features = [
      f
      for f in scores_df["Feature"].head(k).tolist()
      if f is not None and pd.notna(f)
  ]

  return selected_features