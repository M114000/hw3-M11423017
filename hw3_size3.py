import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 讀取 sizes3 資料
# ======================================================
data = pd.read_csv("sizes3.csv")  # 假設資料檔名為 sizes3.csv
X = data.iloc[:, 0:2].values      # 只用特徵
y_true = data.iloc[:, 2].values   # class label 只用於評估

# ======================================================
# 計算 SSE
# ======================================================
def compute_sse(X, labels, centers):
    sse = 0
    for i, c in enumerate(centers):
        sse += np.sum((X[labels == i] - c) ** 2)
    return sse

# ======================================================
# 計算 Accuracy（用最佳對應）
# ======================================================
def clustering_accuracy(y_true, labels_pred):
    labels_unique = np.unique(labels_pred)
    mapping = {}
    for label in labels_unique:
        mask = (labels_pred == label)
        true_labels = y_true[mask]
        if len(true_labels) == 0:
            mapping[label] = 0
        else:
            mapping[label] = np.bincount(true_labels).argmax()
    mapped = np.vectorize(mapping.get)(labels_pred)
    return accuracy_score(y_true, mapped)

# ======================================================
# 計算 Entropy
# ======================================================
def clustering_entropy(labels_pred):
    return entropy(np.bincount(labels_pred))

# ======================================================
# 繪圖函式，每個群中心標上群號
# ======================================================
def plot_cluster(X, labels, title):
    uniq = np.unique(labels)
    plt.figure(figsize=(6,5))
    colors = plt.cm.get_cmap("tab10", len(uniq))
    for i, lab in enumerate(uniq):
        pts = X[labels == lab]
        plt.scatter(pts[:, 0], pts[:, 1], s=80, color=colors(i), label=f"cluster {lab+1}")
        if lab != -1:
            center = pts.mean(axis=0)
            plt.text(center[0], center[1], str(lab+1), fontsize=16,
                     fontweight='bold', ha='center', va='center', color='black')
    plt.title(title)
    plt.legend()
    plt.show()

# ======================================================
# 1. K-means (4 clusters)
# ======================================================
print("\n===== K-means (4 clusters) =====")
start = time.time()
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels_kmeans = kmeans.labels_
end = time.time()
sse_kmeans = compute_sse(X, labels_kmeans, kmeans.cluster_centers_)
acc_kmeans = clustering_accuracy(y_true, labels_kmeans)
ent_kmeans = clustering_entropy(labels_kmeans)
print("Time =", round(end - start, 4), "sec")
print("SSE =", round(sse_kmeans, 4))
print("Accuracy =", round(acc_kmeans, 4))
print("Entropy =", round(ent_kmeans, 4))
plot_cluster(X, labels_kmeans, "K-means Clustering")

# ======================================================
# 2. Hierarchical Clustering (4 clusters)
# ======================================================
print("\n===== Hierarchical Clustering (4 clusters) =====")
start = time.time()
hier = AgglomerativeClustering(n_clusters=4).fit(X)
labels_hier = hier.labels_
end = time.time()
centers_hier = np.array([X[labels_hier == i].mean(axis=0) for i in range(4)])
sse_hier = compute_sse(X, labels_hier, centers_hier)
acc_hier = clustering_accuracy(y_true, labels_hier)
ent_hier = clustering_entropy(labels_hier)
print("Time =", round(end - start, 4), "sec")
print("SSE =", round(sse_hier, 4))
print("Accuracy =", round(acc_hier, 4))
print("Entropy =", round(ent_hier, 4))
plot_cluster(X, labels_hier, "Hierarchical Clustering")

# ======================================================
# DBSCAN (多組參數比較)
# ======================================================
dbscan_params = [
    (0.5, 5),
    (0.7, 5),
    (1.0, 5),
    (0.5, 10),
]

print("\n===== DBSCAN Results =====")

for eps, min_samples in dbscan_params:
    print(f"\n--- DBSCAN eps={eps}, min_samples={min_samples} ---")
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels_db = db.labels_
    end = time.time()

    # DBSCAN 有 -1 (noise)，避免計算錯誤
    valid = labels_db != -1

    if len(np.unique(labels_db)) <= 1:
        print("→ 無法有效分群")
        continue

    # 計算 entropy
    ent_db = clustering_entropy(labels_db[valid])
    
    # 計算 accuracy（排除 noise 點）
    acc_db = clustering_accuracy(y_true[valid], labels_db[valid])

    print("Time =", round(end - start, 4), "sec")
    print("Accuracy =", round(acc_db, 4))
    print("Entropy =", round(ent_db, 4))
    print("Clusters found =", np.unique(labels_db))

    plot_cluster(X, labels_db, f"DBSCAN eps={eps}, min_samples={min_samples}")


