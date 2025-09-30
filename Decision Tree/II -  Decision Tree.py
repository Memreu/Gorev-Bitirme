import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Veri yükleme
data = pd.read_csv("dava_sonuclari.csv")

print("İlk 5 satır:\n", data.head(), "\n")
print("Eksik değer sayısı:\n", data.isnull().sum(), "\n")

# Eksik değer varsa doldurma (örnek: ortalama ile)
data = data.fillna(data.mean(numeric_only=True))

# Kategorik değişkenleri encode et
data_encoded = pd.get_dummies(data, drop_first=True)

# Özellikler (X) ve hedef değişken (y)
X = data_encoded.drop("Outcome", axis=1)
y = data_encoded["Outcome"]

# İsteğe bağlı: Sayısal değişkenleri ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Eğitim veri boyutu:", X_train.shape)
print("Test veri boyutu:", X_test.shape)

# Hiperparametre arama
param_grid = {
    "max_depth": [3, 4, 5, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)

# Modeli kur
dt_model = grid.best_estimator_
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

print("\nModel Değerlendirme Sonuçları:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print("\nDetaylı Sınıflandırma Raporu:\n", classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Kaybet", "Kazan"],
            yticklabels=["Kaybet", "Kazan"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

# Karar ağacı görselleştirme
plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Kaybet", "Kazan"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

# Özellik önemleri
feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
plt.title("Özelliklerin Karar Ağacındaki Önemi")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellik")
plt.show()

print("\nÖzellik Önemleri:\n", feature_importances)
