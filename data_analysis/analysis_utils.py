import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# 1. Genel Yapı, Tipler ve Eksik Veri Analizi
def check_general_structure(df):
  print("=== Veri Tipleri ve Non-Null Bilgisi ===")
  print(df.info())
  print("\n=== Eksik Değer Özeti ===")
  missing = df.isnull().sum()
  missing_pct = (missing / len(df)) * 100
  missing_df = pd.DataFrame(
      {"Eksik Sayısı": missing, "Eksik Oranı (%)": missing_pct}
  )
  print(missing_df[missing_df["Eksik Sayısı"] > 0])
  return missing_df


# 2. İstatistiksel Dağılım
def plot_statistical_distributions(df, numerical_cols):
  for col in numerical_cols:
    if col in df.columns:
      fig, axes = plt.subplots(1, 2, figsize=(12, 4))
      sns.histplot(df[col], kde=True, ax=axes[0], color="blue")
      axes[0].set_title(f"Dağılım (Histogram): {col}")

      sns.boxplot(x=df[col], ax=axes[1], color="orange")
      axes[1].set_title(f"Aykırı Değer (Boxplot): {col}")
      plt.tight_layout()
      plt.show()


# 3. Hedef Değişken (Target) Dağılımı
def plot_target_distribution(df, target_col):
  plt.figure(figsize=(6, 4))
  sns.countplot(data=df, x=target_col, palette="Set2", hue=target_col, legend=False)
  plt.title(f"Hedef Değişken Dağılımı: {target_col}")
  plt.show()


# 4. Coefficient ve Correlation Eğrileri + Target İlişkisi
def analyze_feature_correlations(df, target_col, corr_threshold=0.85):
  corr_matrix = df.select_dtypes(include=[np.number]).corr()

  if target_col in corr_matrix.columns:
    target_corr = (
        corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    )
    print("=== Target ile Korelasyon Sıralaması ===")
    print(target_corr)
    print("-" * 40)
  else:
    target_corr = None

  plt.figure(figsize=(10, 8))
  sns.heatmap(
      corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
  )
  plt.title("Özellikler Arası Korelasyon Matrisi")
  plt.tight_layout()
  plt.show()

  upper_tri = corr_matrix.where(
      np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
  )
  to_drop = [
      col for col in upper_tri.columns if any(abs(upper_tri[col]) > corr_threshold)
  ]
  print(f"⚠️ Yüksek korelasyonlu (çakışan) özellikler: {to_drop}")

  return target_corr, to_drop

# Lineer Modeller İçin
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
    plt.title("Model Katsayıları (Coefficient Eğrisi)")
    plt.axvline(x=0, color="grey", linestyle="--")
    plt.tight_layout()
    plt.show()
    return coef_df
  else:
    print("❌ Bu model 'coef_' niteliğine sahip değil.")
    return None

# Ağaç Modelleri İçin
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
    plt.title("Random Forest - Özellik Önem Dereceleri (Feature Importance)")
    plt.tight_layout()
    plt.show()
    return imp_df
  else:
    print("❌ Bu model 'feature_importances_' niteliğine sahip değil.")
    return None

# 5. Öznitelik Çıkarımı (Feature Extraction) - PCA
def extract_features_pca(X, n_components=2):
  pca = PCA(n_components=n_components)
  X_pca = pca.fit_transform(X.select_dtypes(include=[np.number]))

  pca_df = pd.DataFrame(
      X_pca, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index
  )

  print(f"=== PCA Özellik Çıkarımı ({n_components} Bileşen) ===")
  print(
      f"Açıklanan Toplam Varyans Oranı: {pca.explained_variance_ratio_.sum():.4f}"
  )
  return pca_df, pca


# 6. Öznitelik Çıkarımı (Feature Extraction) - Polinomsal ve Etkileşimler
def extract_polynomial_features(X, degree=2, interaction_only=False):
  num_X = X.select_dtypes(include=[np.number])
  poly = PolynomialFeatures(
      degree=degree, interaction_only=interaction_only, include_bias=False
  )
  X_poly = poly.fit_transform(num_X)
  feature_names = poly.get_feature_names_out(num_X.columns)

  poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)

  print(
      f"=== Polinomsal / Etkileşim Çıkarımı (Derece: {degree}) ==="
      f" Yeni Sütun Sayısı: {poly_df.shape[1]}"
  )
  return poly_df, poly

# 7. Öznitelik Seçimi (Feature Selection)
def select_best_features_anova(X, y, k=5):
  selector = SelectKBest(score_func=f_classif, k=k)
  selector.fit(X, y)
  scores_df = pd.DataFrame(
      {"Feature": X.columns, "Score": selector.scores_}
  ).sort_values(by="Score", ascending=False)

  print(f"=== En İyi {k} Özellik (ANOVA / SelectKBest) ===")
  print(scores_df.head(k))
  return scores_df["Feature"].head(k).tolist()

# 8. Correlation Filtresi
def correlation_filter(X_train, X_test,corr_threshold):
  corr_matrix = X_train.select_dtypes(
    include=[np.number]
  ).corr().abs()

  upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
  )
  to_drop_corr = [
    col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)
  ]

  X_train_reduced = X_train.drop(columns=to_drop_corr)
  X_test_reduced = X_test.drop(columns=to_drop_corr, errors="ignore")
  print(f"9. Korelasyon Filtresi Sonrası Kalan Özellik Sayısı: {X_train_reduced.shape[1]}")
  return X_train_reduced, X_test_reduced

# 9. Outlier Clipping
def apply_iqr_clipping(X_train, X_test, numerical_cols, factor=1.5):
  # 1. Orijinal verileri korumak için kopyalarını oluşturuyoruz
  X_train_clipped = X_train.copy()
  X_test_clipped = X_test.copy()

  outlier_summary = {}

  for col in numerical_cols:
    if col in X_train.columns:
      Q1 = X_train[col].quantile(0.25)
      Q3 = X_train[col].quantile(0.75)
      IQR = Q3 - Q1

      # Sınırları sadece X_train üzerinden hesaplıyoruz
      lower_bound = Q1 - (factor * IQR)
      upper_bound = Q3 + (factor * IQR)

      # Aykırı değer sayılarını raporlama için tutuyoruz
      outliers = X_train[(X_train[col] < lower_bound) | (X_train[col] > upper_bound)]
      outlier_summary[col] = len(outliers)

      # Kırpma (Clipping) işlemlerini DÖNGÜNÜN İÇİNDE her sütun için uyguluyoruz
      X_train_clipped[col] = X_train_clipped[col].clip(lower=lower_bound, upper=upper_bound)
      X_test_clipped[col] = X_test_clipped[col].clip(lower=lower_bound, upper=upper_bound)

  # Raporlama
  print("=== Aykırı Değer (Outlier) Sayıları ===")
  for col, count in outlier_summary.items():
    if count > 0:
      print(f"🔸 {col}: {count} adet aykırı değer bulundu.")

  print(f"8. IQR Clipping İşlemi Tamamlandı (Faktör: {factor}).")
  return X_train_clipped, X_test_clipped, outlier_summary