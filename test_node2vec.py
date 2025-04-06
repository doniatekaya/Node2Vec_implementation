# test_node2vec.py
import pytest
import torch
import os
import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score

# === CONFIGURATION ===
EMBEDDING_PATH = "embeddings_optimized.pt"
LOSS_FILE_PATH = "loss_values.txt"
GRAPH_PATH = "graph.pkl"

@pytest.mark.order(1)
def test_embeddings_file_exists():
    """Vérifie que le fichier d'embeddings a été généré"""
    assert os.path.exists(EMBEDDING_PATH), f"Fichier {EMBEDDING_PATH} introuvable"

@pytest.mark.order(2)
def test_embeddings_shape_and_values():
    """Vérifie la forme et la validité des embeddings"""
    embeddings = torch.load(EMBEDDING_PATH)
    assert isinstance(embeddings, (torch.Tensor, np.ndarray)), \
        "Les embeddings doivent être un tensor pytorch ou un tableau numpy"

    if isinstance(embeddings, np.ndarray):
        assert embeddings.ndim == 2, "Les embeddings numpy doivent être une matrice (2D)"
        assert np.isfinite(embeddings).all(), "Les embeddings contiennent des nan ou Inf"
    else:
        assert embeddings.ndim == 2, "Les embeddings Torch doivent être une matrice (2D)"
        assert torch.isfinite(embeddings).all(), "Les embeddings contiennent des nan ou inf"

@pytest.mark.order(3)
def test_loss_file_exists():
    """Vérifie que le fichier de perte est bien généré."""
    assert os.path.exists(LOSS_FILE_PATH), f"Fichier {LOSS_FILE_PATH} introuvable"

@pytest.mark.order(4)
def test_loss_decreasing():
    """Verifie que la perte diminue au fil des epochs."""
    with open(LOSS_FILE_PATH, "r") as f:
        lines = f.readlines()
    losses = []
    for line in lines:
        if "Loss:" in line:
            parts = line.strip().split("Loss:")
            losses.append(float(parts[1]))

    assert len(losses) >= 2, 
    assert all(np.isfinite(losses)),

    # Optionnel : s'assurer qu'au moins la perte a diminué globalement
    assert losses[-1] <= losses[0], 

@pytest.mark.order(5)
def test_classification_accuracy():
    """teste la précision de classification binaire (infecté vs non infecté)."""
    embeddings = torch.load(EMBEDDING_PATH)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    labels = np.array([G.nodes[n]["infected"] for n in sorted(G.nodes())], dtype=int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n>> Accuracy du classifieur (LogReg): {acc:.4f}")
    assert acc >= 0.5, "l'accuracy est inférieure à 50%  modèle trop aléatoire"

if __name__ == "__main__":
    pytest.main(["-v", __file__])
