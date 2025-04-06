import matplotlib.pyplot as plt
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import networkx as nx
import os
import numpy as np

# === Chargement des embeddings et du graphe ===
embeddings = torch.load("embeddings_optimized.pt")
if isinstance(embeddings, torch.Tensor):
    embeddings = embeddings.detach().cpu().numpy()

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# === Détection de communautés si pas déjà présents ===
if "group" not in list(G.nodes(data=True))[0]:
    try:
        import community as community_louvain  # pip install python-louvain
        partition = community_louvain.best_partition(G)
        nx.set_node_attributes(G, partition, "group")
        print("Groupes détectés automatiquement avec Louvain.")
    except ImportError:
        print("Module python-louvain non installé. Pas de groupes détectés.")
        partition = {}

# === Chargement de la loss (si présente) ===
loss_history = []
if os.path.exists("loss_values.txt"):
    with open("loss_values.txt", "r") as f:
        for line in f:
            if "Loss:" in line:
                val = float(line.strip().split("Loss:")[-1])
                loss_history.append(val)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", marker="o")
    plt.title("Évolution de la perte pendant l'entraînement")
    plt.xlabel("Épochs")
    plt.ylabel("Loss moyenne")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve_reloaded.png")
    plt.show()

# === Visualisation t-SNE ===
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(embeddings)

# Clustering non supervisé (KMeans)
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=cluster_labels, cmap="coolwarm", alpha=0.7)
plt.title("Projection t-SNE des embeddings (KMeans)")
plt.savefig("tsne_miserables.png")
plt.show()

# === PCA pour comparaison ===
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=cluster_labels, cmap="coolwarm", alpha=0.7)
plt.title("Projection PCA des embeddings (KMeans)")
plt.savefig("pca_miserables.png")
plt.show()

# === Calcul ARI / NMI ===
true_labels = [G.nodes[n].get("group") for n in G.nodes if "group" in G.nodes[n]]
if true_labels and len(set(true_labels)) > 1:
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)

    print(f"\n✅ Score ARI: {ari:.4f}")
    print(f"✅ Score NMI: {nmi:.4f}")

    with open("clustering_scores.txt", "w") as f:
        f.write(f"ARI: {ari:.4f}\n")
        f.write(f"NMI: {nmi:.4f}\n")
else:
    print("⚠️ Pas de labels 'group' valides pour évaluer ARI/NMI.")

# === Infos supplémentaires ===
print(f"Nombre de nœuds : {G.number_of_nodes()} | Arêtes : {G.number_of_edges()}")
print(f"Densité du graphe : {nx.density(G):.4f}")
