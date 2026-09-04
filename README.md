# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select Annual Income and Spending Score as features.

2.Standardize the data using StandardScaler.

3.Find the optimal number of clusters using the Elbow Method and Silhouette Score.

4.Apply K-Means clustering with the selected number of clusters (k = 5).

5.Display and visualize the clusters along with their centroids.
## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Atchaya V
RegisterNumber: 212224060031
# ============================================
# K-MEANS CUSTOMER SEGMENTATION
# ============================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================
# 2. Upload CSV file
# ============================================

from google.colab import files

uploaded = files.upload()

# Get uploaded file name
file_name = list(uploaded.keys())[0]

# Load dataset
df = pd.read_csv(file_name)

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

display(df.head())


# ============================================
# 3. Check dataset information
# ============================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================
# 4. Select features
# ============================================

features = ["Annual Income (k$)", "Spending Score (1-100)"]

X = df[features]

print("\nFeatures Used:")
print(features)


# ============================================
# 5. Standardize the data
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================
# 6. Elbow Method
# ============================================

inertia = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(7, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia / SSE")
plt.title("Elbow Method")

plt.grid(True)
plt.show()


# ============================================
# 7. Silhouette Score
# ============================================

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)


plt.figure(figsize=(7, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")

plt.grid(True)
plt.show()


# ============================================
# 8. Apply K-Means
# ============================================

k_final = 5

kmeans = KMeans(
    n_clusters=k_final,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df["Cluster"] = cluster_labels


# ============================================
# 9. Cluster counts
# ============================================

print("\nCluster Counts:")

print(df["Cluster"].value_counts().sort_index())


# ============================================
# 10. Find cluster centers
# ============================================

centers_scaled = kmeans.cluster_centers_

centers_original = scaler.inverse_transform(
    centers_scaled
)

centers_df = pd.DataFrame(
    centers_original,
    columns=features
)

centers_df["Cluster"] = range(k_final)

print("\nCluster Centers:")

display(centers_df.round(2))


# ============================================
# 11. Visualize clusters
# ============================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=70
)

# Plot centroids
plt.scatter(
    centers_df["Annual Income (k$)"],
    centers_df["Spending Score (1-100)"],
    s=250,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend()
plt.grid(True)

plt.show()


# ============================================
# 12. Display final dataset
# ============================================

print("\nFinal Dataset with Cluster Labels:")

display(df.head(10)) 
*/
```

## Output:
<img width="1094" height="819" alt="image" src="https://github.com/user-attachments/assets/580c121a-e3cc-42dd-a109-db39bcf147fc" />

<img width="1102" height="637" alt="image" src="https://github.com/user-attachments/assets/eb179f95-a3c5-4bec-93b9-4a38bd94249e" />

<img width="622" height="470" alt="image" src="https://github.com/user-attachments/assets/679835fe-6dee-4331-b27f-9bdad57238d6" />

<img width="931" height="470" alt="image" src="https://github.com/user-attachments/assets/300e7c26-21f1-4b21-9574-c3f4d211e0f4" />

<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/dfb3a20d-e8fe-49b2-9b38-db5c184d49fb" />

<img width="1004" height="485" alt="image" src="https://github.com/user-attachments/assets/1959dfe9-40c4-40f8-aea8-3a746c1462af" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
