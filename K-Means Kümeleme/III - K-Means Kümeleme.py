import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Veri Hazırlama ve Özellik Seçimi
df = pd.read_csv("dava.csv")
features = ["Case Duration (Days)", "Number of Witnesses", 
            "Legal Fees (USD)", "Number of Evidence Items"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 2. Optimal Küme Sayısının Belirlenmesi (Elbow)
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Yöntemi ile Optimal Küme Sayısı")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("İnertia")
plt.show()


# 3. K-Means Kümeleme Uygulaması
optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters


# 4. Kümeleme Sonuçlarının Görselleştirilmesi ve Yorumlanması
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
plt.title("K-Means Kümeleme Sonuçları (PCA ile 2D)")
plt.xlabel("PCA Bileşeni 1")
plt.ylabel("PCA Bileşeni 2")
plt.colorbar(label="Küme")
plt.show()

print(df.head())
