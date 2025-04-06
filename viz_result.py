# analyse_embeddings.py
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import networkx as nx
import pickle

# Charger embeddings + graphe
embeddings = torch.load("embeddings_optimized.pt")
with open("graph.pkl", "rb") as f:
    G = pickle.load(f)

# Charger loss depuis le fichier texte (loss_values.txt)
loss_history = []
with open("loss_values.txt", "r") as f:
    for line in f:
        if "Loss:" in line:
            val = float(line.strip().split("Loss:")[-1])
            loss_history.append(val)

# Courbe de perte
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

# Visualisation t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_proj = tsne.fit_transform(embeddings)

infected_labels = [G.nodes[n]["infected"] for n in G.nodes]
colors = ["red" if label else "green" for label in infected_labels]

plt.figure(figsize=(8, 6))
plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=colors, alpha=0.7)
plt.title("Projection t-SNE des embeddings Node2Vec")
plt.savefig("tsne_projection_reloaded.png")
plt.show()

# Visualisation PCA pour comparaison
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c=colors, alpha=0.7)
plt.title("Projection PCA des embeddings Node2Vec")
plt.savefig("pca_projection.png")
plt.show()



# Classification supervisée
try:
    X_train, X_test, y_train, y_test = train_test_split(embeddings, infected_labels, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>> Accuracy de classification: {acc:.4f}")

    with open("classification_accuracy.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    print(">> Accuracy sauvegardée dans 'classification_accuracy.txt'")
except Exception as e:
    print("⚠️ Classification non réalisée:", e)
density = nx.density(G)
print(f"--> Densité du graphe: {density:.4f}")
