import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as scipy_entropy


# ---------------------------
# 載入 banana 資料（若沒有則用 make_moons）
# ---------------------------
def load_data():
    try:
        df = pd.read_csv("banana.csv")
        X = df.iloc[:, :2].values
        y = df.iloc[:, 2].values
        print("Loaded banana.csv")
    except:
        print("banana.csv not found → using make_moons instead (banana-like)")
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

    y = LabelEncoder().fit_transform(y)
    return X, y


# ---------------------------
# SSE（忽略 DBSCAN 的 noise = -1）
# ---------------------------
def compute_sse(X, labels):
    sse = 0
    for lab in np.unique(labels):
        if lab == -1:
            continue
        pts = X[labels == lab]
        center = pts.mean(axis=0)
        sse += ((pts - center) ** 2).sum()
    return sse


# ---------------------------
# Accuracy（用 Hungarian 找最佳對應）
# ---------------------------
def clustering_accuracy(y_true, y_pred):
    true_classes = np.unique(y_true)
    pred_classes = np.unique(y_pred)

    C = np.zeros((len(true_classes), len(pred_classes)))

    for i, t in enumerate(true_classes):
        for j, p in enumerate(pred_classes):
            C[i, j] = np.sum((y_true == t) & (y_pred == p))

    row, col = linear_sum_assignment(C.max() - C)
    return C[row, col].sum() / len(y_true)


# ---------------------------
# Entropy
# ---------------------------
def cluster_entropy(y_true, y_pred):
    N = len(y_true)
    ent = 0
    for lab in np.unique(y_pred):
        mask = (y_pred == lab)
        if mask.sum() == 0:
            continue
        vals, cnt = np.unique(y_true[mask], return_counts=True)
        p = cnt / cnt.sum()
        ent += (mask.sum() / N) * scipy_entropy(p, base=2)
    return ent


# ---------------------------
# 畫圖（同群用 + 和 o）
# ---------------------------
def plot_cluster(X, labels, title):
    markers = ['+', 'o']
    uniq = np.unique(labels)

    for i, lab in enumerate(uniq):
        marker = markers[i % len(markers)] if lab != -1 else '.'
        plt.scatter(X[labels == lab, 0], X[labels == lab, 1],
                    marker=marker, s=40, label=f"cluster {lab}")

    plt.title(title)
    plt.legend()
    plt.show()


# ===========================
# 主程式
# ===========================
X, y = load_data()

results = []

# ---------------------------
# 1. K-means
# ---------------------------
start = time.time()
km = KMeans(n_clusters=2, random_state=42)
km_labels = km.fit_predict(X)
t = time.time() - start

results.append(("KMeans", t,
                km.inertia_,  # SSE
                clustering_accuracy(y, km_labels),
                cluster_entropy(y, km_labels)))

plot_cluster(X, km_labels, f"K-means | Acc={results[-1][3]:.3f}")


# ---------------------------
# 2. 階層式分群
# ---------------------------
start = time.time()
hc = AgglomerativeClustering(n_clusters=2, linkage='single')
hc_labels = hc.fit_predict(X)
t = time.time() - start

results.append(("Hierarchical", t,
                compute_sse(X, hc_labels),
                clustering_accuracy(y, hc_labels),
                cluster_entropy(y, hc_labels)))

plot_cluster(X, hc_labels, f"Hierarchical | Acc={results[-1][3]:.3f}")


# ---------------------------
# 3. DBSCAN（多參數實驗）
# ---------------------------
# 設定你要測試的參數組合
dbscan_params = [
    (0.05, 3), (0.1, 3), (0.15, 3),
    (0.05, 5), (0.05, 10), (0.1, 10)
]

print("開始執行 DBSCAN 參數測試...\n")

# 使用迴圈遍歷每一組參數
for eps, ms in dbscan_params:
    # 1. 紀錄時間並訓練模型
    start = time.time()
    # 關鍵：這裡的 eps=eps, min_samples=ms 會將迴圈當前的數值傳入
    db = DBSCAN(eps=eps, min_samples=ms)
    db_labels = db.fit_predict(X)
    t = time.time() - start

    # 2. 計算指標 (SSE, Accuracy, Entropy)
    # 注意：如果全部分成一群或全是雜訊，某些指標可能會報錯或數值很極端，這是正常的
    sse = compute_sse(X, db_labels)
    acc = clustering_accuracy(y, db_labels)
    ent = cluster_entropy(y, db_labels)

    # 3. 將結果存入 results 列表
    # 標籤清楚寫出當前的參數，方便比較
    results.append((f"DBSCAN e={eps},ms={ms}", t, sse, acc, ent))

    # 4. 畫圖 (確保標題顯示當前參數)
    # 建議加上 plt.show() 確保每次迴圈都會把圖畫出來，而不是疊在記憶體中
    plot_cluster(X, db_labels, f"DBSCAN e={eps},ms={ms} | Acc={acc:.3f}")
    plt.show() # 確保這張圖被「印」出來後再跑下一張

# ---------------------------
# 印出總表（終端機）
# ---------------------------
df = pd.DataFrame(results, columns=["Method", "Time", "SSE", "Accuracy", "Entropy"])
print("\n========== Clustering Results ==========\n")
print(df.to_string(index=False))
print("\n========================================\n")
